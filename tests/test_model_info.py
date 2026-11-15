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


# --- Hub, cache and config-shape handling (no real network) -------------------------------

import httpx  # noqa: E402

from autodspark import model_info  # noqa: E402
from autodspark.model_info import hub_token  # noqa: E402

GiB = 1024**3
REPO = "acme/tiny-llm"
TINY = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "num_hidden_layers": 16,
    "hidden_size": 2048,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
}


@pytest.fixture(autouse=True)
def isolated_hf_env(tmp_path, monkeypatch):
    """Keep the real ~/.cache/huggingface, tokens and offline flags out of these tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "XDG_CACHE_HOME",
        "HF_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def fake_hub(monkeypatch):
    """Serve {path: body or status} for httpx.get; records requested URLs."""
    files: dict[str, object] = {}
    seen: list[str] = []

    def get(url, headers=None, params=None, **kw):
        seen.append(url)
        path = url.removeprefix("https://huggingface.co/")
        body = files.get(path, 404)
        req = httpx.Request("GET", url)
        if isinstance(body, int):
            return httpx.Response(body, request=req)
        if isinstance(body, (dict, list)):
            return httpx.Response(200, json=body, request=req)
        return httpx.Response(200, text=str(body), request=req)

    monkeypatch.setattr(model_info.httpx, "get", get)
    get.files, get.seen = files, seen
    return get


def _cache_snapshot(root, repo=REPO, files=None, rev="abc123"):
    repo_dir = root / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / rev
    snap.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(rev)
    for name, body in (files or {}).items():
        p = snap / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body if isinstance(body, str) else json.dumps(body))
    return snap


def _write_config(tmp_path, config, name="m"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# 1. tokens and access errors


def test_hub_token_reads_login_token_files(isolated_hf_env, tmp_path, monkeypatch):
    assert hub_token() is None
    default = isolated_hf_env / ".cache" / "huggingface" / "token"
    default.parent.mkdir(parents=True)
    default.write_text("hf_default\n")
    assert hub_token() == "hf_default"

    hf_home = tmp_path / "hfhome"
    hf_home.mkdir()
    (hf_home / "token").write_text("hf_home_token")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert hub_token() == "hf_home_token"

    custom = tmp_path / "custom-token"
    custom.write_text("hf_custom")
    monkeypatch.setenv("HF_TOKEN_PATH", str(custom))
    assert hub_token() == "hf_custom"

    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hub_token() == "hf_env"
    assert hub_token("hf_explicit") == "hf_explicit"


def test_saved_token_is_sent_to_hub(isolated_hf_env, monkeypatch):
    token_file = isolated_hf_env / ".cache" / "huggingface" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("hf_saved")
    sent = []

    def get(url, headers=None, **kw):
        sent.append((headers or {}).get("Authorization"))
        body = TINY if url.endswith("config.json") else None
        req = httpx.Request("GET", url)
        return httpx.Response(200 if body else 404, json=body, request=req)

    monkeypatch.setattr(model_info.httpx, "get", get)
    inspect_model(REPO)
    assert sent and all(h == "Bearer hf_saved" for h in sent)


def test_401_says_gated_or_misspelled(fake_hub):
    fake_hub.files[f"{REPO}/resolve/main/config.json"] = 401
    with pytest.raises(ModelInfoError) as e:
        inspect_model(REPO)
    msg = str(e.value)
    assert "gated" in msg and "misspelled" in msg and "not exist" in msg
    assert "hf auth login" in msg


# 2. Hugging Face cache and HF_HUB_OFFLINE


def test_offline_reads_hf_cache(tmp_path, monkeypatch, fake_hub):
    cache = tmp_path / "hub-cache"
    _cache_snapshot(
        cache,
        files={
            "config.json": TINY,
            "tokenizer_config.json": {"chat_template": "{{ messages }}"},
            "chat_template.jinja": "<think>{{ messages }}",
            "model.safetensors.index.json": {"metadata": {"total_size": 3 * GiB}},
        },
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    info = inspect_model(REPO)
    assert fake_hub.seen == []
    assert info.num_layers == 16 and info.is_local is False and info.path == REPO
    assert info.weights_gb == pytest.approx(3.0) and info.weights_size_exact is True
    assert info.is_reasoning is True and info.has_chat_template is True


def test_cache_found_via_hf_home(tmp_path, monkeypatch, fake_hub):
    hf_home = tmp_path / "hfhome"
    _cache_snapshot(hf_home / "hub", files={"config.json": TINY})
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "true")
    info = inspect_model(REPO)
    assert info.hidden_size == 2048
    assert info.weights_size_exact is False  # no weights cached, estimated offline
    assert fake_hub.seen == []


def test_cache_used_before_network_online(isolated_hf_env, fake_hub):
    _cache_snapshot(isolated_hf_env / ".cache" / "huggingface" / "hub", files={"config.json": TINY})
    fake_hub.files[f"{REPO}/resolve/main/model.safetensors.index.json"] = {
        "metadata": {"total_size": 2 * GiB}
    }
    info = inspect_model(REPO)
    assert info.num_layers == 16
    assert info.weights_gb == pytest.approx(2.0)
    assert not any(u.endswith("/config.json") for u in fake_hub.seen)


def test_offline_not_cached_is_clear(tmp_path, monkeypatch, fake_hub):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelInfoError, match="HF_HUB_OFFLINE.*isn't in the Hugging Face cache"):
        inspect_model(REPO)
    assert fake_hub.seen == []


# 3. nested / VLM configs


def test_gemma3_text_config_without_vocab_size(tmp_path):
    cfg = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "num_hidden_layers": 34,
            "intermediate_size": 10240,
        },
        "vision_config": {"hidden_size": 1152},
    }
    info = inspect_model(str(_write_config(tmp_path, cfg)))
    assert info.vocab_size == 262208  # transformers' Gemma3TextConfig default
    assert info.num_layers == 34 and info.is_multimodal is True
    assert info.head_dim == 256


def test_llava_text_config_uses_top_level_then_defaults(tmp_path):
    cfg = {
        "architectures": ["LlavaForConditionalGeneration"],
        "model_type": "llava",
        "vocab_size": 32064,
        "text_config": {"model_type": "llama", "max_position_embeddings": 4096},
        "vision_config": {"model_type": "clip_vision_model"},
    }
    info = inspect_model(str(_write_config(tmp_path, cfg)))
    assert info.vocab_size == 32064  # from the top-level config
    assert info.num_layers == 32 and info.hidden_size == 4096  # LlamaConfig defaults
    assert info.max_position == 4096
    assert info.is_multimodal is True


def test_unknown_nested_type_missing_field_names_it(tmp_path):
    cfg = {"model_type": "mystery_vl", "text_config": {"model_type": "mystery", "hidden_size": 8}}
    with pytest.raises(ModelInfoError, match=r"missing \['num_hidden_layers', 'vocab_size'\]"):
        inspect_model(str(_write_config(tmp_path, cfg)))


def test_flat_config_with_vision_config_is_multimodal(tmp_path):
    cfg = {**TINY, "model_type": "qwen2_5_vl", "vision_config": {"depth": 32}}
    assert inspect_model(str(_write_config(tmp_path, cfg))).is_multimodal is True
    cfg = {**TINY, "image_token_id": 151655}
    assert inspect_model(str(_write_config(tmp_path, cfg, "b"))).is_multimodal is True
    assert inspect_model(str(_write_config(tmp_path, TINY, "c"))).is_multimodal is False


# 4. weight sizes


def test_hub_weights_prefer_index_total(fake_hub):
    fake_hub.files[f"{REPO}/resolve/main/config.json"] = TINY
    fake_hub.files[f"{REPO}/resolve/main/model.safetensors.index.json"] = {
        "metadata": {"total_size": 14 * GiB}
    }
    fake_hub.files[f"api/models/{REPO}/tree/main"] = [
        {"path": "consolidated.safetensors", "size": 14 * GiB}
    ]
    info = inspect_model(REPO)
    assert info.weights_gb == pytest.approx(14.0) and info.weights_size_exact is True
    assert not any("/tree/" in u for u in fake_hub.seen)


def test_hub_weights_skip_consolidated_and_subfolders(fake_hub):
    fake_hub.files[f"{REPO}/resolve/main/config.json"] = TINY
    fake_hub.files[f"api/models/{REPO}/tree/main"] = [
        {"path": "model.safetensors", "size": 4 * GiB},
        {"path": "consolidated.safetensors", "size": 4 * GiB},
        {"path": "original/model.safetensors", "size": 4 * GiB},
        {"path": "original", "type": "directory"},
    ]
    assert inspect_model(REPO).weights_gb == pytest.approx(4.0)


def test_local_weights_skip_consolidated(tmp_path):
    d = _write_config(tmp_path, TINY)
    (d / "model.safetensors").write_bytes(b"x" * 1000)
    (d / "consolidated.safetensors").write_bytes(b"x" * 5000)
    info = inspect_model(str(d))
    assert info.weights_gb == pytest.approx(1000 / GiB, abs=0.01)
    assert info.weights_size_exact is True


# 5. chat_template.jinja on the Hub


def test_hub_fetches_chat_template_jinja(fake_hub):
    fake_hub.files[f"{REPO}/resolve/main/config.json"] = TINY
    fake_hub.files[f"{REPO}/resolve/main/tokenizer_config.json"] = {"bos_token": "<s>"}
    fake_hub.files[f"{REPO}/resolve/main/chat_template.jinja"] = "{% if enable_thinking %}x"
    info = inspect_model(REPO)
    assert info.has_chat_template is True and info.is_reasoning is True


def test_hub_missing_chat_template_is_fine(fake_hub):
    fake_hub.files[f"{REPO}/resolve/main/config.json"] = TINY
    info = inspect_model(REPO)
    assert info.has_chat_template is False


# 6. odd config shapes and context-length keys


def test_list_intermediate_size_takes_max(tmp_path):
    cfg = {**TINY, "intermediate_size": [8192, 16384, 8192]}
    info = inspect_model(str(_write_config(tmp_path, cfg)))
    assert info.intermediate_size == 16384


def test_list_intermediate_size_when_estimating(tmp_path):
    cfg = {**TINY, "intermediate_size": [8192, 16384]}
    info = inspect_model(str(_write_config(tmp_path, cfg)))
    assert info.weights_size_exact is False and info.weights_gb > 0


@pytest.mark.parametrize(
    "patch",
    [
        {"hidden_size": "big"},
        {"num_hidden_layers": [1, 2]},
        {"text_config": [1, 2]},
        {"num_attention_heads": {"a": 1}},
    ],
)
def test_bad_config_shapes_raise_model_info_error(tmp_path, patch):
    with pytest.raises(ModelInfoError):
        inspect_model(str(_write_config(tmp_path, {**TINY, **patch})))


def test_config_not_an_object(tmp_path):
    d = tmp_path / "list"
    d.mkdir()
    (d / "config.json").write_text("[1, 2]")
    with pytest.raises(ModelInfoError, match="JSON object"):
        inspect_model(str(d))


@pytest.mark.parametrize("key", ["seq_length", "n_positions", "max_sequence_length"])
def test_context_length_fallback_keys(tmp_path, key):
    cfg = {k: v for k, v in TINY.items() if k != "max_position_embeddings"}
    cfg[key] = 4096
    assert inspect_model(str(_write_config(tmp_path, cfg))).max_position == 4096
