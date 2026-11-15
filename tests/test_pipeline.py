import json
import os
import subprocess
import sys

import pytest
from rich.console import Console

from autodspark import pipeline as pl
from autodspark.hardware import simulated
from autodspark.model_info import inspect_model
from autodspark.planner import make_plan
from autodspark.procs import StepFailed
from autodspark.runtime import Runtime


@pytest.fixture
def setup(model_dir, tmp_path):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    rt = Runtime.resolve(home=tmp_path / "home")
    console = Console(record=True, width=200)
    return plan, rt, tmp_path / "run", console


def test_dry_run_prints_every_stage_and_records_nothing(setup):
    plan, rt, workdir, console = setup
    pl.Pipeline(plan, rt, workdir, console, dry_run=True).run()
    out = console.export_text()
    for needle in (
        "regenerate-responses",
        "prepare-data",
        "speculators train",
        "launch_vllm.py train",
        "--speculator-type dspark",
    ):
        assert needle in out
    assert not (workdir / "state.json").exists()


def test_offline_plan_has_hidden_stage(model_dir, tmp_path):
    plan = make_plan(
        inspect_model(str(model_dir("qwen3-0.6b"))), simulated("2x80"), "quick", mode="offline"
    )
    p = pl.Pipeline(plan, Runtime.resolve(home=tmp_path), tmp_path / "r", Console(), dry_run=True)
    assert "hidden" in p.stages()


def test_stage_hashes_change_downstream_only(setup):
    plan, *_ = setup
    before = pl.stage_hashes(plan)
    changed = plan.model_copy(deep=True)
    changed.train.epochs += 1
    after = pl.stage_hashes(changed)
    assert before["regen"] == after["regen"]
    assert before["prepare"] == after["prepare"]
    assert before["train"] != after["train"]
    assert before["eval"] != after["eval"]


def test_resume_skips_done_stages(setup, monkeypatch):
    plan, rt, workdir, console = setup
    ran = []
    p = pl.Pipeline(plan, rt, workdir, console)
    for stage in p.stages():
        monkeypatch.setattr(p, f"_stage_{stage}", lambda s=stage: ran.append(s) or {})
    p.run()
    assert ran == p.stages()

    ran.clear()
    p2 = pl.Pipeline(plan, rt, workdir, console)
    for stage in p2.stages():
        monkeypatch.setattr(p2, f"_stage_{stage}", lambda s=stage: ran.append(s) or {})
    p2.run()
    assert ran == []  # everything already done

    plan.train.epochs += 1  # training inputs changed: redo train and after
    p3 = pl.Pipeline(plan, rt, workdir, console)
    for stage in p3.stages():
        monkeypatch.setattr(p3, f"_stage_{stage}", lambda s=stage: ran.append(s) or {})
    p3.run()
    assert ran == ["train", "eval", "report"]


def test_failed_stage_is_recorded(setup, monkeypatch):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)

    def boom():
        raise StepFailed("regen broke")

    monkeypatch.setattr(p, "_stage_regen", boom)
    with pytest.raises(StepFailed):
        p.run()
    state = json.loads((workdir / "state.json").read_text())
    assert state["stages"]["regen"]["status"] == "failed"


def test_oom_retry_halves_anchors(setup, monkeypatch):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)
    seen = []

    def fake_exec(cmd, log_name, grace=None):
        anchors = int(cmd.argv[cmd.argv.index("--max-anchors") + 1])
        seen.append(anchors)
        if len(seen) < 3:
            log = workdir / "logs" / log_name
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as f:
                f.write("torch.OutOfMemoryError: CUDA out of memory.\n")
            raise StepFailed("train failed", log)
        ckpt = workdir / "checkpoints" / "checkpoint_best"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text("{}")

    monkeypatch.setattr(p, "_exec", fake_exec)
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    info = p._stage_train()
    start = plan.draft.max_anchors
    assert seen == [start, start // 2, start // 4]
    assert info["max_anchors"] == start // 4
    assert info["drafter"].endswith("checkpoint_best")


def test_non_oom_failure_is_not_retried(setup, monkeypatch):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)
    calls = []

    def fake_exec(cmd, log_name, grace=None):
        calls.append(1)
        log = workdir / "logs" / log_name
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ValueError: something else\n")
        raise StepFailed("train failed", log)

    monkeypatch.setattr(p, "_exec", fake_exec)
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    with pytest.raises(StepFailed):
        p._stage_train()
    assert len(calls) == 1


def test_find_drafter_prefers_best(tmp_path):
    # Speculators names epochs by integer; epoch3_step500 is a readability alias.
    for name in ("1", "3", "2", "epoch3_step500"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    assert pl.find_drafter(tmp_path).name == "3"
    (tmp_path / "checkpoint_best").mkdir()
    (tmp_path / "checkpoint_best" / "config.json").write_text("{}")
    assert pl.find_drafter(tmp_path).name == "checkpoint_best"
    assert pl.find_drafter(tmp_path / "missing") is None


def test_changed_inputs_move_old_outputs_aside(setup):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)
    out = workdir / "checkpoints"
    p._claim_dir(out, "aaa")
    (out / "checkpoint_1").mkdir()
    p._claim_dir(out, "aaa")  # same inputs (e.g. resuming after a crash): keep
    assert (out / "checkpoint_1").exists()
    p._claim_dir(out, "bbb")  # plan changed: move aside, start empty
    assert out.exists() and not any(out.iterdir())
    stale = list(workdir.glob("checkpoints.stale-*"))
    assert len(stale) == 1 and (stale[0] / "checkpoint_1").exists()


def test_regen_asks_only_for_missing_conversations(setup, monkeypatch):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)
    src = plan.data.sources[0]
    out = p._regen_file(src.slug)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 300 conversations done before a crash; two rows (turns) for each
    rows = [
        json.dumps({"primary_id": f"c{i}", "id": f"c{i}-{t}"}) for i in range(300) for t in range(2)
    ]
    out.write_text("\n".join(rows) + "\n")
    asked = []

    def fake_exec(cmd, log_name, grace=None):
        asked.append(int(cmd.argv[cmd.argv.index("--limit") + 1]))
        with out.open("a") as f:
            f.write("\n".join(json.dumps({"primary_id": f"n{i}"}) for i in range(asked[-1])))

    monkeypatch.setattr(p, "_exec", fake_exec)
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    info = p._stage_regen()
    assert asked == [src.limit - 300]
    assert info["conversations"][src.slug] == src.limit


def test_count_regenerated_counts_conversations(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text('{"primary_id": "a"}\n{"primary_id": "a"}\n{"primary_id": "b"}\nnot json\n')
    assert pl.count_regenerated(f) == 2
    assert pl.count_regenerated(tmp_path / "missing.jsonl") == 0


def test_old_oom_in_log_is_not_blamed(setup, monkeypatch):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)
    log = workdir / "logs" / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("torch.OutOfMemoryError: from an earlier run\n")
    calls = []

    def fake_exec(cmd, log_name, grace=None):
        calls.append(1)
        with log.open("a") as f:
            f.write("ValueError: a different problem\n")
        raise StepFailed("train failed", log)

    monkeypatch.setattr(p, "_exec", fake_exec)
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    with pytest.raises(StepFailed, match="train failed"):
        p._stage_train()
    assert len(calls) == 1


def test_server_failure_is_not_retried_as_oom(setup, monkeypatch):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)

    class Broken:
        def __enter__(self):
            raise StepFailed("vLLM (hidden-state extraction) exited during startup")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(p, "_server", lambda *a, **k: Broken())
    monkeypatch.setattr(p, "_exec", lambda *a, **k: pytest.fail("train should not start"))
    with pytest.raises(StepFailed, match="exited during startup"):
        p._stage_train()


def test_forced_stage_starts_from_clean_outputs(setup, monkeypatch):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)
    for stage in p.stages():
        if stage != "train":
            monkeypatch.setattr(p, f"_stage_{stage}", lambda: {})
    ckpt = workdir / "checkpoints"

    def fake_train(*a, **k):
        (ckpt / "0").mkdir()
        (ckpt / "0" / "config.json").write_text("{}")

    monkeypatch.setattr(p, "_exec", fake_train)
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    p._claim_dir(ckpt, p.hashes["train"])
    (ckpt / "0").mkdir()
    (ckpt / "0" / "config.json").write_text("{}")
    p.run(from_stage="train")
    stale = list(workdir.glob("checkpoints.stale-*"))
    assert len(stale) == 1 and (stale[0] / "0").exists()  # old run kept, not resumed


def test_find_drafter_keeps_best_symlink(tmp_path):
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "config.json").write_text("{}")
    (tmp_path / "checkpoint_best").symlink_to("0")
    assert pl.find_drafter(tmp_path) == (tmp_path / "checkpoint_best").absolute()


def _write_jsonl(path, n):
    path.write_text("".join(json.dumps({"prompt": f"q{i}"}) + "\n" for i in range(n)))
    return path


def test_stage_hashes_follow_file_contents(model_dir, tmp_path):
    model = model_dir("qwen3-0.6b")
    (model / "model.safetensors").write_bytes(b"x" * 100)
    data = _write_jsonl(tmp_path / "mine.jsonl", 50)
    plan = make_plan(inspect_model(str(model)), simulated("1x24"), "quick", data_files=[str(data)])
    base = pl.stage_hashes(plan)
    assert pl.stage_hashes(plan) == base  # stable: resuming an unchanged run skips stages

    _write_jsonl(data, 51)  # data file edited in place
    edited = pl.stage_hashes(plan)
    assert edited["regen"] != base["regen"]

    (model / "model.safetensors").write_bytes(b"y" * 200)  # fine-tune re-saved in place
    resaved = pl.stage_hashes(plan)
    assert resaved["regen"] != edited["regen"]

    cfg = json.loads((model / "config.json").read_text())
    cfg["rope_theta"] = 1
    (model / "config.json").write_text(json.dumps(cfg))
    assert pl.stage_hashes(plan)["regen"] != resaved["regen"]


def test_hub_model_is_fingerprinted_by_id(setup):
    plan, *_ = setup
    hub = plan.model.model_copy(update={"path": "Qwen/Qwen3-8B", "is_local": False})
    assert pl.model_fingerprint(hub) == "Qwen/Qwen3-8B"


def test_old_or_broken_state_file_does_not_crash(setup, monkeypatch):
    plan, rt, workdir, console = setup
    workdir.mkdir(parents=True)
    # state.json from an older version: hashes no longer match, so stages rerun
    old = {"stages": {"regen": {"status": "done", "hash": "0123456789abcdef", "info": {}}}}
    (workdir / "state.json").write_text(json.dumps(old))
    p = pl.Pipeline(plan, rt, workdir, console)
    ran = []
    for stage in p.stages():
        monkeypatch.setattr(p, f"_stage_{stage}", lambda s=stage: ran.append(s) or {})
    p.run()
    assert ran == p.stages()
    (workdir / "state.json").write_text("{not json")
    assert pl.State(workdir / "state.json").get("regen") is None


def test_unknown_from_stage_lists_valid_stages(setup):
    plan, rt, workdir, console = setup
    with pytest.raises(ValueError, match="stages are: regen, prepare, train"):
        pl.Pipeline(plan, rt, workdir, console).run(from_stage="hidden")


def _oom_exec(workdir):
    def fake_exec(cmd, log_name, grace=None):
        log = workdir / "logs" / log_name
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write("torch.OutOfMemoryError: CUDA out of memory.\n")
        raise StepFailed("train failed", log)

    return fake_exec


def test_oom_message_after_retries(setup, monkeypatch):
    plan, rt, workdir, console = setup
    plan.draft.max_anchors = 4096
    p = pl.Pipeline(plan, rt, workdir, console, max_oom_retries=2)
    monkeypatch.setattr(p, "_exec", _oom_exec(workdir))
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    with pytest.raises(StepFailed) as e:
        p._stage_train()
    msg = str(e.value)
    assert "after 2 retries" in msg and "last tried 1024" in msg
    assert "other processes using the GPU" in msg and "DataLoader" in msg
    assert '"--gradient-checkpointing", "--num-workers", "2"' in msg


def test_oom_message_when_nothing_was_shrunk(setup, monkeypatch):
    plan, rt, workdir, console = setup
    plan.draft.max_anchors = 300
    p = pl.Pipeline(plan, rt, workdir, console)
    monkeypatch.setattr(p, "_exec", _oom_exec(workdir))
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    with pytest.raises(StepFailed) as e:
        p._stage_train()
    msg = str(e.value)
    assert "even after shrinking" not in msg and "retries" not in msg
    assert "max_anchors=300" in msg and "too small to shrink" in msg


def test_online_train_clears_leftover_hidden_states(setup, monkeypatch):
    plan, rt, workdir, console = setup
    assert plan.layout.mode == "online"
    p = pl.Pipeline(plan, rt, workdir, console)
    leftover = workdir / "hs_stream" / "old" / "hs_0.safetensors"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("from a crashed run")

    def fake_train(cmd, log_name, grace=None):
        assert not any((workdir / "hs_stream").iterdir())
        (workdir / "checkpoints" / "0").mkdir(parents=True)
        (workdir / "checkpoints" / "0" / "config.json").write_text("{}")

    monkeypatch.setattr(p, "_exec", fake_train)
    monkeypatch.setattr(p, "_server", lambda *a, **k: __import__("contextlib").nullcontext())
    p._stage_train()
    assert not leftover.exists()


def test_eval_without_drafter_fails_clearly(setup):
    plan, rt, workdir, console = setup
    p = pl.Pipeline(plan, rt, workdir, console)
    p.state.data["stages"]["train"] = {"info": {"drafter": str(workdir / "gone")}}
    assert p._drafter() is None  # recorded path no longer exists
    with pytest.raises(StepFailed, match="No trained drafter"):
        p._stage_eval()


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX only")
def test_workdir_lock_is_exclusive_and_dies_with_its_process(tmp_path):
    lock = tmp_path / "run" / ".lock"
    with pl.WorkdirLock(lock), pytest.raises(pl.WorkdirBusy, match=f"pid {os.getpid()}"):
        pl.WorkdirLock(lock).__enter__()
    with pl.WorkdirLock(lock):  # released on exit
        pass
    # A holder that is killed (no cleanup at all) must not block the next run.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR);"
            "fcntl.flock(fd, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(60)",
            str(lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "locked"
    with pytest.raises(pl.WorkdirBusy):
        pl.WorkdirLock(lock).__enter__()
    holder.kill()
    holder.wait()
    with pl.WorkdirLock(lock):
        pass


def test_workdir_lock_runs_unlocked_when_locking_is_unsupported(tmp_path, monkeypatch):
    """ENOLCK & co. (e.g. Lustre without flock) must not be reported as "busy" forever."""
    import errno
    import fcntl

    from autodspark.pipeline import WorkdirLock

    def no_locks(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    with WorkdirLock(tmp_path / ".lock"):
        pass
