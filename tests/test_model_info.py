import json

import pytest

from autodspark.model_info import ModelInfoError, inspect_model


def test_dense_model(model_dir):
    info = inspect_model(str(model_dir("qwen3-0.6b")))
    assert info.num_layers == 28
    assert info.hidden_size == 1024
    assert info.vocab_size == 151936
    assert info.is_moe is False
    assert info.is_multimodal is False
    assert info.weights_size_exact is True
    assert info.weights_gb == pytest.approx(1.4, abs=0.01)
    assert info.is_reasoning is True  # template mentions enable_thinking
    assert info.has_chat_template is True
    assert info.is_local is True


def test_multimodal_uses_text_config(model_dir):
    info = inspect_model(str(model_dir("gemma-vl")))
    assert info.is_multimodal is True
    assert info.num_layers == 34
    assert info.hidden_size == 2560
    assert info.model_type == "gemma3_text"


def test_moe_detected(model_dir):
    info = inspect_model(str(model_dir("qwen3-30b-a3b")))
    assert info.is_moe is True
    assert info.num_experts == 128


def test_mla_custom_code_and_quant(model_dir):
    info = inspect_model(str(model_dir("mla-custom")))
    assert info.uses_mla is True
    assert info.needs_remote_code is True
    assert info.quant_method == "fp8"
    assert info.is_moe is True


def test_no_chat_template(model_dir):
    info = inspect_model(str(model_dir("llama-3.2-1b", chat_template=None)))
    assert info.has_chat_template is False
    assert info.is_reasoning is False


def test_size_estimated_without_weights(model_dir):
    d = model_dir("qwen3-8b")
    (d / "model.safetensors.index.json").unlink()
    info = inspect_model(str(d))
    assert info.weights_size_exact is False
    assert 10 < info.weights_gb < 25  # rough estimate for an 8B bf16 model


def test_gguf_rejected(tmp_path):
    d = tmp_path / "gguf-model"
    d.mkdir()
    (d / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    with pytest.raises(ModelInfoError, match="GGUF"):
        inspect_model(str(d))


def test_missing_config(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ModelInfoError, match="config.json"):
        inspect_model(str(tmp_path / "empty"))


def test_missing_fields(tmp_path):
    d = tmp_path / "weird"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "weird"}))
    with pytest.raises(ModelInfoError, match="missing"):
        inspect_model(str(d))


def test_not_a_repo_id():
    with pytest.raises(ModelInfoError, match="neither"):
        inspect_model("definitely-not-a-folder-or-repo")


def test_hub_endpoint_respects_mirror(monkeypatch):
    from autodspark.model_info import hub_endpoint

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert hub_endpoint() == "https://huggingface.co"
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert hub_endpoint() == "https://hf-mirror.com"
