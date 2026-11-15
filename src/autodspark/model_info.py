"""Read a target model's config and work out what the planner needs.

Works on a local HuggingFace-format directory or a Hub repo id. Only small
JSON files are read; weights are never loaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

GiB = 1024**3

# Keys different architectures use for the routed-expert count.
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def hub_endpoint() -> str:
    """Respect HF_ENDPOINT (e.g. https://hf-mirror.com) like huggingface_hub does."""
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def hub_token(explicit: str | None = None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class ModelInfoError(RuntimeError):
    """The model can't be used for DSpark training as given."""


class ModelInfo(BaseModel):
    path: str
    name: str
    model_type: str
    architectures: list[str] = []
    num_layers: int
    hidden_size: int
    intermediate_size: int | None = None
    vocab_size: int
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    max_position: int | None = None
    is_moe: bool = False
    num_experts: int | None = None
    is_multimodal: bool = False
    uses_mla: bool = False
    is_hybrid: bool = False
    quant_method: str | None = None
    weights_gb: float
    weights_size_exact: bool = False
    is_reasoning: bool = False
    has_chat_template: bool = False
    needs_remote_code: bool = False
    is_local: bool = False

    @property
    def size_label(self) -> str:
        return f"{self.weights_gb:.1f} GB"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _hub_json(repo_id: str, filename: str, token: str | None = None) -> dict[str, Any] | None:
    """Fetch a small JSON file from the Hub without needing huggingface_hub."""
    url = f"{hub_endpoint()}/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    except httpx.HTTPError as e:
        raise ModelInfoError(f"Could not reach the Hugging Face Hub for {repo_id}: {e}") from e
    if r.status_code == 404:
        return None
    if r.status_code in (401, 403):
        raise ModelInfoError(
            f"{repo_id} is gated or private. Run `huggingface-cli login` or set HF_TOKEN."
        )
    r.raise_for_status()
    return r.json()


def _hub_weights_gb(repo_id: str, token: str | None = None) -> float | None:
    """Total size of *.safetensors files in a Hub repo, if we can get it."""
    url = f"{hub_endpoint()}/api/models/{repo_id}/tree/main"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(url, headers=headers, params={"recursive": "true"}, timeout=30)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    total = sum(
        item.get("size", 0)
        for item in r.json()
        if isinstance(item, dict) and str(item.get("path", "")).endswith(".safetensors")
    )
    return total / GiB if total else None


def _local_weights_gb(model_dir: Path) -> float | None:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        total = _read_json(index).get("metadata", {}).get("total_size")
        if total:
            return float(total) / GiB
    total = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))
    return total / GiB if total else None


def _estimate_params(text_cfg: dict[str, Any], num_experts: int | None) -> float:
    """Rough parameter count from the config, used only when file sizes are unknown."""
    h = text_cfg.get("hidden_size", 0)
    n_layers = text_cfg.get("num_hidden_layers", 0)
    vocab = text_cfg.get("vocab_size", 0)
    inter = text_cfg.get("moe_intermediate_size") or text_cfg.get("intermediate_size") or 4 * h
    experts = num_experts or 1
    per_layer = 4 * h * h + 3 * h * inter * experts
    embed = vocab * h * (1 if text_cfg.get("tie_word_embeddings") else 2)
    return float(n_layers * per_layer + embed)


def _detect_reasoning(tokenizer_cfg: dict[str, Any] | None, chat_template: str | None) -> bool:
    template = chat_template or ""
    if tokenizer_cfg and isinstance(tokenizer_cfg.get("chat_template"), str):
        template += tokenizer_cfg["chat_template"]
    return any(tag in template for tag in ("enable_thinking", "<think>", "reasoning_content"))


def inspect_model(model: str, hf_token: str | None = None) -> ModelInfo:
    """Build a ModelInfo for a local directory or a Hub repo id."""
    hf_token = hub_token(hf_token)
    local = Path(model).expanduser()
    if local.suffix == ".gguf" or (local.is_dir() and any(local.glob("*.gguf"))):
        raise ModelInfoError(
            "GGUF models can't be trained against: DSpark training reads the target's hidden "
            "states, which needs the original Hugging Face weights (safetensors) loaded in vLLM. "
            "Point --model at the original HF checkpoint of this model."
        )

    chat_template: str | None = None
    if local.is_dir():
        cfg_path = local / "config.json"
        if not cfg_path.exists():
            raise ModelInfoError(f"No config.json in {local}. Is this a Hugging Face model folder?")
        config = _read_json(cfg_path)
        tok_path = local / "tokenizer_config.json"
        tokenizer_cfg = _read_json(tok_path) if tok_path.exists() else None
        jinja = local / "chat_template.jinja"
        if jinja.exists():
            chat_template = jinja.read_text(encoding="utf-8")
        weights_gb = _local_weights_gb(local)
        path, is_local = str(local.resolve()), True
    elif local.exists():
        raise ModelInfoError(f"{local} is a file. Pass the model folder instead.")
    else:
        if model.count("/") != 1:
            raise ModelInfoError(f"{model} is neither a local folder nor a Hub repo id (org/name).")
        config = _hub_json(model, "config.json", hf_token)
        if config is None:
            raise ModelInfoError(f"Could not find config.json for {model} on the Hub.")
        tokenizer_cfg = _hub_json(model, "tokenizer_config.json", hf_token)
        weights_gb = _hub_weights_gb(model, hf_token)
        path, is_local = model, False

    text_cfg = config.get("text_config") or config.get("llm_config") or config
    is_multimodal = text_cfg is not config

    missing = [k for k in ("num_hidden_layers", "hidden_size", "vocab_size") if k not in text_cfg]
    if missing:
        raise ModelInfoError(
            f"config.json is missing {missing}; this architecture isn't supported."
        )

    num_experts = next((int(text_cfg[k]) for k in _MOE_KEYS if text_cfg.get(k)), None)
    layer_types = [str(t).lower() for t in text_cfg.get("layer_types") or []]
    model_type = str(text_cfg.get("model_type") or config.get("model_type") or "unknown")
    quant = config.get("quantization_config") or text_cfg.get("quantization_config") or {}

    exact = weights_gb is not None
    if weights_gb is None:
        bytes_per_param = 1 if quant else 2
        weights_gb = _estimate_params(text_cfg, num_experts) * bytes_per_param / GiB

    return ModelInfo(
        path=path,
        name=Path(path).name if is_local else model.split("/")[-1],
        model_type=model_type,
        architectures=list(config.get("architectures") or []),
        num_layers=int(text_cfg["num_hidden_layers"]),
        hidden_size=int(text_cfg["hidden_size"]),
        intermediate_size=text_cfg.get("intermediate_size"),
        vocab_size=int(text_cfg["vocab_size"]),
        num_attention_heads=text_cfg.get("num_attention_heads"),
        num_kv_heads=text_cfg.get("num_key_value_heads"),
        head_dim=text_cfg.get("head_dim"),
        max_position=text_cfg.get("max_position_embeddings"),
        is_moe=num_experts is not None and num_experts > 1,
        num_experts=num_experts,
        is_multimodal=is_multimodal,
        uses_mla=bool(text_cfg.get("kv_lora_rank")),
        is_hybrid=any(t for t in layer_types if "mamba" in t or "linear" in t or "ssm" in t)
        or "mamba" in model_type,
        quant_method=quant.get("quant_method") if isinstance(quant, dict) else None,
        weights_gb=round(weights_gb, 2),
        weights_size_exact=exact,
        is_reasoning=_detect_reasoning(tokenizer_cfg, chat_template),
        has_chat_template=bool(
            chat_template or (tokenizer_cfg and tokenizer_cfg.get("chat_template"))
        ),
        needs_remote_code="auto_map" in config,
        is_local=is_local,
    )
