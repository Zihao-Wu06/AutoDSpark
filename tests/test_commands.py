import json
from pathlib import Path

import pytest

from autodspark import commands
from autodspark.hardware import simulated
from autodspark.model_info import inspect_model
from autodspark.planner import make_plan
from autodspark.runtime import Runtime


@pytest.fixture
def rt(tmp_path):
    return Runtime.resolve(home=tmp_path / "home")


def _plan(model_dir, name="qwen3-0.6b", gpus="1x24", **kw):
    return make_plan(inspect_model(str(model_dir(name))), simulated(gpus), "quick", **kw)


def _after(argv, flag, n=1):
    i = argv.index(flag)
    return argv[i + 1 : i + 1 + n]


def test_layer_ids_identical_on_both_sides(model_dir, rt, tmp_path):
    """The bug community scripts hit: vLLM and training must use the same layers."""
    plan = _plan(model_dir)
    ids = [str(i) for i in plan.draft.target_layer_ids]
    server = commands.extract_server(plan, rt, tmp_path / "hs")
    train = commands.train(plan, rt, tmp_path / "prep", tmp_path / "ckpt")
    assert _after(server.argv, "--target-layer-ids", len(ids)) == ids
    assert _after(train.argv, "--target-layer-ids", len(ids)) == ids


def test_shared_gpu_online(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    server = commands.extract_server(plan, rt, tmp_path / "hs")
    train = commands.train(plan, rt, tmp_path / "prep", tmp_path / "ckpt")
    assert server.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert train.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert _after(server.argv, "--gpu-memory-utilization") == [str(plan.layout.vllm_mem_util)]
    assert _after(train.argv, "--on-missing") == ["generate"]
    assert _after(train.argv, "--vllm-endpoint") == ["http://127.0.0.1:8101/v1"]
    assert train.argv[1:4] == ["-m", "speculators", "train"]  # single GPU: no torchrun


def test_multi_gpu_uses_torchrun(model_dir, rt, tmp_path):
    plan = _plan(model_dir, "qwen3-8b", "4x80")
    train = commands.train(plan, rt, tmp_path / "prep", tmp_path / "ckpt")
    assert "torch.distributed.run" in train.argv
    assert _after(train.argv, "--nproc_per_node") == [str(len(plan.layout.train_gpus))]
    assert train.env["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, plan.layout.train_gpus))


def test_offline_training_reads_cache(model_dir, rt, tmp_path):
    plan = _plan(model_dir, gpus="2x80", mode="offline")
    with pytest.raises(ValueError):
        commands.train(plan, rt, tmp_path / "prep", tmp_path / "ckpt")
    train = commands.train(plan, rt, tmp_path / "prep", tmp_path / "ckpt", hs_cache=tmp_path / "hs")
    assert _after(train.argv, "--on-missing") == ["warn"]
    assert "--vllm-endpoint" not in train.argv


def test_dspark_flags(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    argv = commands.train(plan, rt, tmp_path / "p", tmp_path / "c", max_anchors=256).argv
    assert _after(argv, "--speculator-type") == ["dspark"]
    assert _after(argv, "--max-anchors") == ["256"]
    assert _after(argv, "--draft-vocab-size") == ["32000"]
    assert json.loads(_after(argv, "--loss-fn")[0]) == {"ce": 0.1, "tv": 0.9}
    assert "--save-best" in argv


def test_prepare_writes_token_freq_for_reduced_vocab(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    argv = commands.prepare(plan, rt, [tmp_path / "a.jsonl", tmp_path / "b.jsonl"], tmp_path).argv
    assert "--no-skip-token-freq" in argv
    assert argv.count("--data") == 2


def test_regen_uses_chat_endpoint(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    src = plan.data.sources[0]
    argv = commands.regen(plan, rt, src, tmp_path / "out.jsonl").argv
    assert _after(argv, "--endpoint") == ["http://127.0.0.1:8100/v1/chat/completions"]
    assert _after(argv, "--limit") == [str(src.limit)]
    assert "--resume" in argv


def test_trust_remote_code_only_when_needed(model_dir, rt, tmp_path):
    plain = _plan(model_dir)
    assert "--trust-remote-code" not in commands.regen_server(plain, rt).argv
    custom = _plan(model_dir, "mla-custom", "2x80")
    assert "--trust-remote-code" in commands.regen_server(custom, rt).argv
    assert "--trust-remote-code" in commands.train(custom, rt, tmp_path, tmp_path).argv


def test_eval_server_uses_explicit_speculative_config(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    base = commands.eval_server(plan, rt, None)
    spec = commands.eval_server(plan, rt, tmp_path / "ckpt")
    assert base.argv[4] == spec.argv[4] == plan.model.path
    assert "--speculative-config" not in base.argv
    cfg = json.loads(_after(spec.argv, "--speculative-config")[0])
    assert cfg == {"method": "dspark", "model": str(tmp_path / "ckpt"), "num_speculative_tokens": 8}
    assert _after(spec.argv, "--served-model-name") == ["target"]


def test_serve_commands(model_dir, tmp_path):
    plan = _plan(model_dir)  # local model folder
    cmds = commands.serve_commands(plan, Path("/ckpt/checkpoint_best"))
    assert len(cmds) == 1  # the short form doesn't work for local targets
    assert '"method": "dspark"' in cmds[0][1]
    hub = plan.model_copy(deep=True)
    hub.model.is_local = False
    hub.model.path = "Qwen/Qwen3-0.6B"
    cmds = commands.serve_commands(hub, Path("/ckpt/checkpoint_best"))
    assert cmds[1][1] == "vllm serve /ckpt/checkpoint_best"


def test_shell_rendering_quotes(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    text = commands.train(plan, rt, tmp_path / "p", tmp_path / "c").shell()
    assert text.startswith("CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ")
    assert '\'{"ce":0.1,"tv":0.9}\'' in text


def test_markov_rank_zero_disables_confidence_markov(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    plan.draft.markov_rank = 0
    argv = commands.train(plan, rt, tmp_path, tmp_path).argv
    assert "--no-confidence-head-with-markov" in argv
    assert "--confidence-head-with-markov" not in argv


def test_regen_limit_override(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    argv = commands.regen(plan, rt, plan.data.sources[0], tmp_path / "o.jsonl", limit=7).argv
    assert _after(argv, "--limit") == ["7"]


def test_prepare_uses_every_regenerated_row(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    assert "--max-samples" not in commands.prepare(plan, rt, [tmp_path / "a.jsonl"], tmp_path).argv


def test_train_extra_args_pass_through(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    plan.train.extra_args = ["--gradient-checkpointing", "--num-workers", "2"]
    argv = commands.train(plan, rt, tmp_path, tmp_path).argv
    assert _after(argv, "--num-workers") == ["2"]
    assert "--gradient-checkpointing" in argv


def test_user_gpu_uuids_become_physical_indices(model_dir, rt, tmp_path):
    """vLLM parses CUDA_VISIBLE_DEVICES as integers, so UUIDs are mapped to PCI indices."""
    plan = _plan(model_dir, gpus="2x80")
    for g, u in zip(plan.hardware.gpus, ["GPU-aaaa", "GPU-bbbb"], strict=True):
        g.cuda_id = u
    for cmd in (
        commands.regen_server(plan, rt),
        commands.extract_server(plan, rt, tmp_path / "hs"),
        commands.train(plan, rt, tmp_path, tmp_path),
        commands.eval_server(plan, rt, None),
    ):
        assert cmd.env["CUDA_VISIBLE_DEVICES"] == ",".join(str(i) for i in cmd.gpus)
        assert cmd.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_vllm_memory_follows_plan(model_dir, rt, tmp_path):
    plan = _plan(model_dir)
    assert _after(commands.regen_server(plan, rt).argv, "--gpu-memory-utilization") == ["0.9"]
    assert _after(commands.eval_server(plan, rt, None).argv, "--gpu-memory-utilization") == ["0.85"]
    plan.layout.regen_mem_util, plan.layout.eval_mem_util = 0.55, 0.5
    assert _after(commands.regen_server(plan, rt).argv, "--gpu-memory-utilization") == ["0.55"]
    assert _after(commands.eval_server(plan, rt, None).argv, "--gpu-memory-utilization") == ["0.5"]


def test_serve_commands_quote_paths_with_spaces(model_dir):
    import shlex

    plan = _plan(model_dir, gpus="2x80")
    plan.model.path = "/models/my model"
    plan.layout.regen_tp = 2
    drafter = Path("/work dir/checkpoint_best")
    argv = shlex.split(commands.serve_commands(plan, drafter)[0][1])
    assert argv[:3] == ["vllm", "serve", "/models/my model"]
    assert _after(argv, "--tensor-parallel-size") == ["2"]
    assert json.loads(_after(argv, "--speculative-config")[0])["model"] == str(drafter)
    hub = plan.model_copy(deep=True)
    hub.model.is_local = False
    hub.model.path = "Qwen/Qwen3-0.6B"
    short = commands.serve_commands(hub, drafter)[1][1]
    assert shlex.split(short) == ["vllm", "serve", str(drafter), "--tensor-parallel-size", "2"]
