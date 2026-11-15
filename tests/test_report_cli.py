import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autodspark import cli, report
from autodspark.cli import app
from autodspark.hardware import simulated
from autodspark.model_info import inspect_model
from autodspark.planner import Plan, make_plan

runner = CliRunner()

EVAL = {
    "baseline": {
        "num_requests": 40,
        "tokens_per_second": 52.1,
        "by_category": {"code": 50.0, "chat": 54.0},
    },
    "dspark": {
        "num_requests": 40,
        "tokens_per_second": 118.4,
        "acceptance_length": 3.41,
        "acceptance_rate": 0.34,
        "by_category": {"code": 131.0, "chat": 109.0},
    },
    "compare": {
        "speedup": 2.27,
        "acceptance_length": 3.41,
        "acceptance_rate": 0.34,
        "by_category": {
            "code": {"baseline_tps": 50.0, "dspark_tps": 131.0},
            "chat": {"baseline_tps": 54.0, "dspark_tps": 109.0},
        },
    },
}


def test_report_render(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    text = report.render(plan, Path("/runs/x/checkpoints/checkpoint_best"), EVAL)
    assert "**2.27x faster**" in text
    assert "| code | 50.0 | 131.0 | 2.62x |" in text
    assert '"model": "/runs/x/checkpoints/checkpoint_best"' in text
    assert "target layers [2, 14, 25]" in text


def test_report_without_eval(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    text = report.render(plan, None, None)
    assert "## Training setup" in text
    assert "## Result" not in text


def test_cli_plan_and_status(model_dir, tmp_path):
    model = model_dir("llama-3.2-1b")
    workdir = tmp_path / "run"
    r = runner.invoke(
        app, ["plan", "--model", str(model), "--gpus", "2x24", "--workdir", str(workdir)]
    )
    assert r.exit_code == 0, r.output
    assert (workdir / "plan.yaml").exists()
    assert Plan.load(workdir / "plan.yaml").layout.train_gpus == [1]

    r = runner.invoke(app, ["status", "--workdir", str(workdir)])
    assert r.exit_code == 0, r.output
    assert "pending" in r.output


def test_cli_dry_run_from_workdir(model_dir, tmp_path):
    model = model_dir("qwen3-0.6b")
    workdir = tmp_path / "run"
    runner.invoke(app, ["plan", "--model", str(model), "--gpus", "1x80", "--workdir", str(workdir)])
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "speculator-type dspark" in r.output.replace("\n", " ")


def test_cli_gpus_flag_needs_dry_run(model_dir, tmp_path):
    r = runner.invoke(app, ["run", "--model", str(model_dir("qwen3-0.6b")), "--gpus", "1x24"])
    assert r.exit_code == 2


def test_cli_bad_model(tmp_path):
    (tmp_path / "empty").mkdir()
    r = runner.invoke(app, ["plan", "--model", str(tmp_path / "empty"), "--gpus", "1x24"])
    assert r.exit_code == 2
    assert "config.json" in r.output


def test_cli_serve_without_drafter(model_dir, tmp_path):
    workdir = tmp_path / "run"
    runner.invoke(
        app,
        [
            "plan",
            "--model",
            str(model_dir("qwen3-0.6b")),
            "--gpus",
            "1x24",
            "--workdir",
            str(workdir),
        ],
    )
    r = runner.invoke(app, ["serve", "--workdir", str(workdir)])
    assert r.exit_code == 2
    assert "No trained drafter" in r.output


def test_cli_run_keeps_edited_plan(model_dir, tmp_path):
    model = model_dir("qwen3-0.6b")
    workdir = tmp_path / "run"
    runner.invoke(app, ["plan", "--model", str(model), "--gpus", "1x80", "--workdir", str(workdir)])
    p = Plan.load(workdir / "plan.yaml")
    p.train.epochs = 9
    p.save(workdir / "plan.yaml")
    r = runner.invoke(
        app,
        ["run", "--model", str(model), "--workdir", str(workdir), "--dry-run", "--samples", "5"],
    )
    assert r.exit_code == 0, r.output
    assert "ignored" in r.output
    assert "--epochs 9" in r.output.replace("\n", " ")
    assert Plan.load(workdir / "plan.yaml").train.epochs == 9


def test_cli_rejects_broken_plan_edit(model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_is_linux", lambda: True)
    model = model_dir("qwen3-0.6b")
    workdir = tmp_path / "run"
    runner.invoke(app, ["plan", "--model", str(model), "--gpus", "1x80", "--workdir", str(workdir)])
    p = Plan.load(workdir / "plan.yaml")
    p.draft.target_layer_ids = [2, 28]
    p.save(workdir / "plan.yaml")
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--yes"])
    assert r.exit_code == 2
    assert "target_layer_ids" in r.output


def test_report_explains_a_slow_quick_run(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    slow = {**EVAL, "compare": {**EVAL["compare"], "speedup": 0.88}}
    text = report.render(plan, Path("/ckpt"), slow)
    assert "This was a `quick` run" in text
    assert "doesn't speed generation up yet" in text
    assert "Small target models" in text


def test_report_headline_for_slowdown(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    slow = {**EVAL, "compare": {**EVAL["compare"], "speedup": 0.876}}
    text = report.render(plan, Path("/ckpt"), slow)
    assert "**0.88x the baseline speed**" in text
    assert "faster**" not in text


def _plan(model, workdir, *extra):
    r = runner.invoke(
        app, ["plan", "--model", str(model), "--gpus", "1x80", "--workdir", str(workdir), *extra]
    )
    assert r.exit_code == 0, r.output
    return r


def test_report_wording_for_shared_gpus(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    assert plan.layout.shared_gpu
    assert "(vLLM and training share one GPU)" in report.render(plan, None, None)
    plan.layout.vllm_gpus = plan.layout.train_gpus = [0, 1]
    text = report.render(plan, None, None)
    assert "share one GPU" not in text
    assert "each of the 2 GPUs ran a vLLM replica and a training rank" in text


def test_cli_resume_refuses_a_different_model(model_dir, tmp_path):
    workdir = tmp_path / "run"
    _plan(model_dir("qwen3-0.6b"), workdir)
    other = model_dir("qwen3-8b")
    r = runner.invoke(app, ["run", "-m", str(other), "--workdir", str(workdir), "--dry-run"])
    assert r.exit_code == 2
    out = r.output.replace("\n", " ")
    assert "--replan" in out and "--model" in out
    r = runner.invoke(
        app,
        ["run", "-m", str(other), "--workdir", str(workdir), "--dry-run", "--replan"]
        + ["--gpus", "1x80"],
    )
    assert r.exit_code == 0, r.output
    assert "qwen3-8b" in r.output


def test_cli_resume_refuses_a_different_preset(model_dir, tmp_path):
    model = model_dir("qwen3-0.6b")
    workdir = tmp_path / "run"
    _plan(model, workdir, "--preset", "quick")
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--preset", "standard", "--dry-run"])
    assert r.exit_code == 2
    assert "--preset standard" in r.output
    # not passing --preset (or passing the same one) resumes the saved plan
    r = runner.invoke(app, ["run", "-m", str(model), "--workdir", str(workdir), "--dry-run"])
    assert r.exit_code == 0, r.output


def test_cli_resume_notes_ignored_trust_remote_code(model_dir, tmp_path):
    workdir = tmp_path / "run"
    _plan(model_dir("qwen3-0.6b"), workdir)
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--trust-remote-code", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "--trust-remote-code" in r.output and "ignored" in r.output


@pytest.mark.parametrize("stage", ["hidden", "trian"])
def test_cli_bad_from_stage_is_a_clean_error(model_dir, tmp_path, stage):
    workdir = tmp_path / "run"
    _plan(model_dir("qwen3-0.6b"), workdir)
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--from-stage", stage, "--dry-run"])
    assert r.exit_code == 2
    assert r.exception is None or isinstance(r.exception, SystemExit)
    assert "regen, prepare, train, eval, report" in " ".join(r.output.split())


def test_cli_resume_hint_is_shell_quoted(model_dir, tmp_path):
    workdir = tmp_path / "my runs" / "a"
    r = _plan(model_dir("qwen3-0.6b"), workdir)
    assert f"autodspark run --workdir '{workdir}'" in r.output.replace("\n", "")


def test_cli_serve_with_missing_recorded_drafter(model_dir, tmp_path):
    workdir = tmp_path / "run"
    _plan(model_dir("qwen3-0.6b"), workdir)
    state = {"stages": {"train": {"status": "done", "info": {"drafter": str(tmp_path / "gone")}}}}
    (workdir / "state.json").write_text(json.dumps(state))
    r = runner.invoke(app, ["serve", "--workdir", str(workdir)])
    assert r.exit_code == 2
    assert "is gone" in r.output
    r = runner.invoke(app, ["serve", "--workdir", str(tmp_path / "nothing")])
    assert r.exit_code == 2 and "No plan.yaml" in r.output


def test_cli_refuses_real_work_off_linux(model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_is_linux", lambda: False)
    workdir = tmp_path / "run"
    _plan(model_dir("qwen3-0.6b"), workdir)  # planning works anywhere
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--dry-run"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--yes"])
    assert r.exit_code == 2 and "needs Linux" in r.output
    r = runner.invoke(app, ["setup", "--home", str(tmp_path / "home")])
    assert r.exit_code == 2 and "needs Linux" in r.output


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX only")
def test_cli_second_run_on_same_workdir_is_refused(model_dir, tmp_path, monkeypatch):
    from autodspark.pipeline import Paths, WorkdirLock

    monkeypatch.setattr(cli, "_is_linux", lambda: True)
    workdir = tmp_path / "run"
    _plan(model_dir("qwen3-0.6b"), workdir)
    with WorkdirLock(Paths(workdir).lock):
        r = runner.invoke(app, ["run", "--workdir", str(workdir), "--yes"])
        assert r.exit_code == 1
        assert "Another autodspark run" in r.output
        # read-only commands and previews don't need the lock
        assert runner.invoke(app, ["status", "--workdir", str(workdir)]).exit_code == 0
        assert runner.invoke(app, ["run", "--workdir", str(workdir), "--dry-run"]).exit_code == 0
