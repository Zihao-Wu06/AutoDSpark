import json

import pytest

from autodspark.hardware import simulated
from autodspark.model_info import inspect_model
from autodspark.planner import (
    Plan,
    PlanError,
    default_target_layers,
    make_plan,
)


@pytest.fixture
def info(model_dir):
    return lambda name, **kw: inspect_model(str(model_dir(name, **kw)))


def test_target_layers_match_speculators_default():
    assert default_target_layers(36) == [2, 18, 33]
    assert default_target_layers(28) == [2, 14, 25]
    layers = default_target_layers(4)
    assert all(0 < i < 4 for i in layers) and len(set(layers)) == len(layers)


def test_small_model_single_gpu_shares_it(info):
    p = make_plan(info("qwen3-0.6b"), simulated("1x24"), "standard")
    assert p.layout.mode == "online"
    assert p.layout.shared_gpu
    assert p.layout.vllm_gpus == p.layout.train_gpus == [0]
    assert 0.1 <= p.layout.vllm_mem_util < 0.5
    assert p.draft.num_layers == 3
    assert p.draft.draft_vocab_size == 32000
    assert not p.errors


def test_two_gpus_split_vllm_and_training(info):
    p = make_plan(info("qwen3-8b"), simulated("2x80"), "standard")
    assert p.layout.mode == "online"
    assert not p.layout.shared_gpu
    assert p.layout.vllm_gpus == [0]
    assert p.layout.train_gpus == [1]
    assert p.layout.regen_gpus == [0, 1]
    assert p.layout.regen_dp == 2
    assert p.draft.num_layers == 5


def test_eight_gpus_give_half_to_vllm(info):
    p = make_plan(info("qwen3-8b"), simulated("8x80"), "standard")
    assert len(p.layout.vllm_gpus) == 4
    assert len(p.layout.train_gpus) == 4
    assert set(p.layout.vllm_gpus).isdisjoint(p.layout.train_gpus)


def test_big_model_uses_tensor_parallel(info):
    p = make_plan(info("qwen3-30b-a3b"), simulated("4x40"), "standard")
    assert p.layout.vllm_tp == 2
    assert len(p.layout.vllm_gpus) % 2 == 0
    assert p.layout.train_gpus  # training still gets GPUs


def test_model_too_big(info):
    with pytest.raises(PlanError, match="needs about"):
        make_plan(info("qwen3-30b-a3b"), simulated("1x24"), "standard")


def test_no_gpus(info):
    from autodspark.hardware import Hardware

    with pytest.raises(PlanError, match="No NVIDIA GPU"):
        make_plan(info("qwen3-0.6b"), Hardware(), "standard")


def test_tight_gpu_shrinks_drafter_with_warning(info):
    p = make_plan(info("qwen3-8b"), simulated("1x40"), "standard")
    assert not p.errors
    assert p.draft.num_layers <= 5
    assert any("lowered" in w for w in p.warnings) or p.draft.max_anchors < 3072


def test_offline_needs_disk(info):
    hw = simulated("1x24", disk_free_gb=50)
    p = make_plan(info("qwen3-8b"), hw, "standard", samples=50000)
    assert p.layout.mode == "offline" or p.errors
    if p.layout.mode == "offline":
        assert p.estimates.offline_disk_gb > 50
        assert any("Not enough disk" in e for e in p.errors)


def test_user_data_then_public_top_up(info, tmp_path):
    f = tmp_path / "mine.jsonl"
    f.write_text("\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(300)) + "\n")
    p = make_plan(info("qwen3-0.6b"), simulated("1x24"), "quick", data_files=[str(f)])
    assert [s.kind for s in p.data.sources] == ["file", "preset"]
    assert p.data.sources[0].limit == 300
    assert p.data.sources[1].limit == 700  # quick preset = 1000 samples
    assert sum(s.limit for s in p.data.sources) == p.data.max_samples


def test_user_data_only(info, tmp_path):
    f = tmp_path / "mine.jsonl"
    f.write_text("\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(5000)) + "\n")
    p = make_plan(
        info("qwen3-0.6b"), simulated("1x24"), "quick", data_files=[str(f)], public_data="none"
    )
    assert len(p.data.sources) == 1
    assert p.data.sources[0].limit == 1000


def test_missing_data_file(info):
    with pytest.raises(PlanError, match="not found"):
        make_plan(info("qwen3-0.6b"), simulated("1x24"), data_files=["/nope.jsonl"])


def test_unknown_preset_and_dataset(info):
    with pytest.raises(PlanError):
        make_plan(info("qwen3-0.6b"), simulated("1x24"), "huge")
    with pytest.raises(PlanError, match="Unknown public dataset"):
        make_plan(info("qwen3-0.6b"), simulated("1x24"), public_data="wikipedia")


def test_bad_target_layers(info):
    with pytest.raises(PlanError, match="target_layer_ids"):
        make_plan(info("qwen3-0.6b"), simulated("1x24"), target_layer_ids=[2, 2, 99])


def test_mla_custom_model_plan(info):
    p = make_plan(info("mla-custom"), simulated("2x80"), "standard")
    assert p.draft.draft_vocab_size == 32000  # 102400 > 64000
    assert p.trust_remote_code
    assert any("speculators#1175" in w for w in p.warnings)  # MLA note
    assert any("fp8" in w for w in p.warnings)


def test_multimodal_note(info):
    p = make_plan(info("gemma-vl"), simulated("1x80"), "standard")
    assert any("Vision-language" in w for w in p.warnings)


def test_force_offline(info):
    p = make_plan(info("qwen3-0.6b"), simulated("2x80"), "quick", mode="offline")
    assert p.layout.mode == "offline"
    assert p.layout.train_gpus == [0, 1]


def test_plan_roundtrip(info, tmp_path):
    p = make_plan(info("qwen3-0.6b"), simulated("1x24"), "quick")
    path = tmp_path / "plan.yaml"
    p.save(path)
    assert path.read_text().startswith("# AutoDSpark training plan")
    q = Plan.load(path)
    assert q.model_dump() == p.model_dump()


def _short_context(model_dir, max_pos):
    d = model_dir("llama-3.2-1b")
    cfg = json.loads((d / "config.json").read_text())
    cfg["max_position_embeddings"] = max_pos
    (d / "config.json").write_text(json.dumps(cfg))
    return inspect_model(str(d))


def test_short_context_model_fits_every_server(model_dir):
    """2k-context models: seq + 1 and prompt + answer must fit the context."""
    p = make_plan(_short_context(model_dir, 2048), simulated("1x24"), "quick")
    assert p.data.seq_len + 1 <= 2048
    assert p.data.regen_max_model_len <= 2048
    assert p.data.regen_max_tokens <= p.data.regen_max_model_len // 2
    assert not p.problems()


def test_8k_context_full_preset_leaves_room_for_prompts(model_dir):
    p = make_plan(_short_context(model_dir, 8192), simulated("1x80"), "full")
    assert p.data.seq_len == 8191
    assert p.data.regen_max_model_len == 8192
    assert p.data.regen_max_tokens == 4096


def test_final_layer_is_not_a_valid_target(info):
    with pytest.raises(PlanError, match="target_layer_ids"):
        make_plan(info("qwen3-0.6b"), simulated("1x24"), target_layer_ids=[2, 14, 28])


def test_data_file_must_be_jsonl(info, tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("hello\n")
    with pytest.raises(PlanError, match="jsonl"):
        make_plan(info("qwen3-0.6b"), simulated("1x24"), data_files=[str(f)])


def test_same_file_names_get_unique_outputs(info, tmp_path):
    files = []
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        f = tmp_path / sub / "data.jsonl"
        f.write_text(json.dumps({"prompt": "q"}) + "\n")
        files.append(str(f))
    p = make_plan(info("qwen3-0.6b"), simulated("1x24"), "quick", data_files=files)
    slugs = [s.slug for s in p.data.sources]
    assert len(set(slugs)) == len(slugs)


def test_problems_catch_bad_edits(info):
    p = make_plan(info("qwen3-0.6b"), simulated("1x24"), "quick")
    assert p.problems() == []
    bad = p.model_copy(deep=True)
    bad.draft.target_layer_ids = [2, 28]
    bad.data.seq_len = 50000
    bad.layout.train_gpus = [3]
    probs = " ".join(bad.problems())
    assert "target_layer_ids" in probs
    assert "context" in probs
    assert "GPUs [3]" in probs


def test_kv_estimate_uses_head_dim(model_dir):
    from autodspark.planner import kv_bytes_per_token

    d = model_dir("qwen3-0.6b")
    cfg = json.loads((d / "config.json").read_text())
    cfg["head_dim"] = 128  # Qwen3-0.6B: not hidden / heads (= 64)
    (d / "config.json").write_text(json.dumps(cfg))
    m = inspect_model(str(d))
    assert kv_bytes_per_token(m) == 2 * 28 * 8 * 128 * 2


def test_medium_preset(info):
    p = make_plan(info("qwen3-8b"), simulated("1x80"), "medium")
    assert p.data.max_samples == 10000
    assert p.train.epochs == 3
    assert p.data.seq_len == 2048
