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
from pydantic import BaseModel, ValidationError

GiB = 1024**3

# Keys different architectures use for the routed-expert count.
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def hub_endpoint() -> str:
    """Respect HF_ENDPOINT (e.g. https://hf-mirror.com) like huggingface_hub does."""
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _hf_home() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "huggingface"


def _token_files() -> list[Path]:
    """Where `huggingface-cli login` / `hf auth login` may have stored a token, in order."""
    files = []
    if os.environ.get("HF_TOKEN_PATH"):
        files.append(Path(os.environ["HF_TOKEN_PATH"]).expanduser())
    files.append(_hf_home() / "token")
    files.append(Path.home() / ".cache" / "huggingface" / "token")
    return files


def hub_token(explicit: str | None = None) -> str | None:
    """Explicit token, then HF_TOKEN / HUGGING_FACE_HUB_TOKEN, then the saved login token."""
    token = explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    for f in _token_files():
        try:
            saved = f.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if saved:
            return saved
    return None


def hub_offline() -> bool:
    """HF_HUB_OFFLINE=1 means: never touch the network, like huggingface_hub."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")


def hub_cache_dir() -> Path:
    """The huggingface_hub download cache (HF_HUB_CACHE, else $HF_HOME/hub)."""
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    return Path(env).expanduser() if env else _hf_home() / "hub"


def _cached_snapshot(repo_id: str) -> Path | None:
    """The cached snapshot folder for repo_id's main revision, if it was downloaded."""
    repo = hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo / "snapshots"
    try:
        rev = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        rev = ""
    if rev and (snapshots / rev).is_dir():
        return snapshots / rev
    # No refs/main (e.g. downloaded by commit hash): use the only snapshot if unambiguous.
    found = [d for d in snapshots.iterdir() if d.is_dir()] if snapshots.is_dir() else []
    return found[0] if len(found) == 1 else None


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


def _parse_json(text: str, where: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ModelInfoError(f"{where} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ModelInfoError(f"{where} should hold a JSON object, got {type(data).__name__}.")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ModelInfoError(f"Could not read {path}: {e}") from e
    return _parse_json(text, str(path))


def _hub_file(repo_id: str, filename: str, token: str | None = None) -> str | None:
    """Fetch a small text file from the Hub without needing huggingface_hub. None on 404."""
    url = f"{hub_endpoint()}/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    except httpx.HTTPError as e:
        raise ModelInfoError(f"Could not reach the Hugging Face Hub for {repo_id}: {e}") from e
    if r.status_code == 404:
        return None
    if r.status_code in (401, 403):
        hint = (
            "the token in use was rejected or has no access to it"
            if token
            else "no Hugging Face token was found"
        )
        raise ModelInfoError(
            f"The Hub refused access to {repo_id} (HTTP {r.status_code}); {hint}. The repo may be "
            "gated or private, or the id may be misspelled or not exist. Check the id at "
            f"{hub_endpoint()}/{repo_id}. For a gated or private repo, accept its license on that "
            "page, then run `hf auth login` (or `huggingface-cli login`) or set HF_TOKEN."
        )
    try:
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ModelInfoError(f"Could not fetch {filename} for {repo_id} from the Hub: {e}") from e
    return r.text


def _hub_json(repo_id: str, filename: str, token: str | None = None) -> dict[str, Any] | None:
    """Fetch a small JSON file from the Hub without needing huggingface_hub."""
    text = _hub_file(repo_id, filename, token)
    return None if text is None else _parse_json(text, f"{filename} of {repo_id}")


def _is_weight_file(name: str) -> bool:
    """Top-level *.safetensors that make up the HF checkpoint.

    Skips consolidated*.safetensors (Mistral's native-format copy shipped next to the shards)
    and anything in a subfolder (e.g. gpt-oss's original/ copy), which would double count.
    """
    return name.endswith(".safetensors") and "/" not in name and not name.startswith("consolidated")


def _index_total_gb(index: dict[str, Any] | None) -> float | None:
    meta = index.get("metadata") if isinstance(index, dict) else None
    total = meta.get("total_size") if isinstance(meta, dict) else None
    try:
        total = float(total) if total is not None and not isinstance(total, bool) else None
    except (TypeError, ValueError):
        return None
    return total / GiB if total and total > 0 else None


def _hub_weights_gb(repo_id: str, token: str | None = None) -> float | None:
    """Total size of the top-level *.safetensors files in a Hub repo, if we can get it."""
    url = f"{hub_endpoint()}/api/models/{repo_id}/tree/main"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        items = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(items, list):
        return None
    total = 0
    for item in items:
        if isinstance(item, dict) and _is_weight_file(str(item.get("path", ""))):
            size = item.get("size")
            total += size if isinstance(size, int) else 0
    return total / GiB if total else None


def _local_weights_gb(model_dir: Path) -> float | None:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        try:
            gb = _index_total_gb(_read_json(index))
        except ModelInfoError:
            gb = None
        if gb:
            return gb
    total = sum(
        p.stat().st_size for p in model_dir.glob("*.safetensors") if _is_weight_file(p.name)
    )
    return total / GiB if total else None


def _estimate_params(text_cfg: dict[str, Any], num_experts: int | None) -> float:
    """Rough parameter count from the config, used only when file sizes are unknown."""
    h = text_cfg.get("hidden_size") or 0
    n_layers = text_cfg.get("num_hidden_layers") or 0
    vocab = text_cfg.get("vocab_size") or 0
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


# Keys a composite (vision-language) config nests its language model config under.
_TEXT_CONFIG_KEYS = ("text_config", "llm_config", "language_config")

# Top-level keys that mean the model takes images even when its text fields are flat
# (Qwen2.5-VL, LLaVA-style configs).
_VISION_KEYS = (
    "vision_config",
    "vision_tower",
    "mm_vision_tower",
    "visual",
    "image_token_id",
    "image_token_index",
    "vision_start_token_id",
)

# Nested sub-configs are saved as a diff against their class defaults, so a field equal to the
# transformers default is simply left out (Gemma 3's text_config has no vocab_size; LLaVA-7B's
# text_config has no hidden_size or num_hidden_layers). These are those documented defaults,
# for the text model types where we've checked them.
_TEXT_DEFAULTS: dict[str, dict[str, int]] = {
    "llama": {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 2048,
    },
    "mistral": {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
    },
    "qwen2": {
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 22016,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "max_position_embeddings": 32768,
    },
    "gemma": {
        "vocab_size": 256000,
        "hidden_size": 3072,
        "intermediate_size": 24576,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 256,
        "max_position_embeddings": 8192,
    },
    "gemma2": {
        "vocab_size": 256000,
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 8192,
    },
    "gemma3_text": {
        "vocab_size": 262208,
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 131072,
    },
}

# Text model type a composite config uses when its text_config doesn't say.
_DEFAULT_TEXT_TYPE = {"llava": "llama", "gemma3": "gemma3_text", "paligemma": "gemma"}

_REQUIRED = ("num_hidden_layers", "hidden_size", "vocab_size")
_CONTEXT_KEYS = ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")


def _to_int(value: Any, field: str, *, list_max: bool = False) -> int | None:
    """Coerce a config value to int, or raise a ModelInfoError naming the field."""
    if value is None:
        return None
    if list_max and isinstance(value, (list, tuple)) and value:
        # Gemma 3n and similar give one value per layer; the largest bounds memory.
        return max(_to_int(v, field) or 0 for v in value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ModelInfoError(
        f"config.json has an unexpected value for {field}: {value!r} (expected an integer). "
        "Fix config.json or point --model at the original Hugging Face checkpoint."
    )


class _Config:
    """Field lookup: the text config first, then the top-level config, then known defaults."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        text_cfg: Any = next((config[k] for k in _TEXT_CONFIG_KEYS if config.get(k)), config)
        if not isinstance(text_cfg, dict):
            raise ModelInfoError(
                f"config.json has an unexpected text config ({type(text_cfg).__name__}, "
                "expected an object). Fix config.json or use the original HF checkpoint."
            )
        self.text = text_cfg
        self.nested = text_cfg is not config
        parent_type = str(config.get("model_type") or "")
        self.model_type = str(
            text_cfg.get("model_type") or _DEFAULT_TEXT_TYPE.get(parent_type) or parent_type
        )
        self.model_type = self.model_type or "unknown"

    def raw(self, key: str) -> Any:
        if self.text.get(key) is not None:
            return self.text[key]
        if self.nested and self.config.get(key) is not None:
            return self.config[key]
        return None

    def get(self, key: str) -> Any:
        value = self.raw(key)
        if value is None:
            value = _TEXT_DEFAULTS.get(self.model_type, {}).get(key)
        return value


def _load_config(raw: dict[str, Any]) -> tuple[_Config, dict[str, Any]]:
    """Resolve the fields the planner needs; returns the lookup and the resolved values."""
    cfg = _Config(raw)
    missing = [k for k in _REQUIRED if cfg.get(k) is None]
    if missing:
        where = "text_config" if cfg.nested else "config.json"
        raise ModelInfoError(
            f"config.json is missing {missing} (model_type {cfg.model_type!r}), and there's no "
            f"known default for it. Add the value(s) to {where}, copying them from the model "
            "card or the original checkpoint, or this architecture isn't supported."
        )
    vals: dict[str, Any] = {k: _to_int(cfg.get(k), k) for k in _REQUIRED}
    for k in ("intermediate_size", "moe_intermediate_size"):
        vals[k] = _to_int(cfg.get(k), k, list_max=True)
    for k in ("num_attention_heads", "num_key_value_heads", "head_dim"):
        vals[k] = _to_int(cfg.get(k), k)
    ctx_key = next((k for k in _CONTEXT_KEYS if cfg.raw(k) is not None), None)
    ctx = cfg.raw(ctx_key) if ctx_key else cfg.get("max_position_embeddings")
    vals["max_position_embeddings"] = _to_int(ctx, ctx_key or "max_position_embeddings")
    vals["tie_word_embeddings"] = bool(cfg.raw("tie_word_embeddings"))
    for k, v in vals.items():
        if isinstance(v, int) and not isinstance(v, bool) and v < 0:
            raise ModelInfoError(f"config.json has a negative {k} ({v}).")
    return cfg, vals


def _is_multimodal(cfg: _Config) -> bool:
    return cfg.nested or any(cfg.config.get(k) is not None for k in _VISION_KEYS)


def _offline_error(repo_id: str) -> ModelInfoError:
    return ModelInfoError(
        f"HF_HUB_OFFLINE is set and {repo_id} isn't in the Hugging Face cache "
        f"({hub_cache_dir()}). Download it first (`hf download {repo_id}`), pass the local "
        "model folder instead, or unset HF_HUB_OFFLINE."
    )


def inspect_model(model: str, hf_token: str | None = None) -> ModelInfo:
    """Build a ModelInfo for a local directory or a Hub repo id."""
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
        config, tokenizer_cfg, chat_template, weights_gb = _from_hub(model, hub_token(hf_token))
        path, is_local = model, False

    cfg, vals = _load_config(config)
    text_cfg = cfg.text
    try:
        num_experts = next((_to_int(text_cfg[k], k) for k in _MOE_KEYS if text_cfg.get(k)), None)
        layer_types = [str(t).lower() for t in text_cfg.get("layer_types") or []]
    except TypeError as e:
        raise ModelInfoError(f"config.json has an unexpected shape: {e}") from e
    model_type = cfg.model_type
    quant = config.get("quantization_config") or text_cfg.get("quantization_config") or {}

    exact = weights_gb is not None
    if weights_gb is None:
        bytes_per_param = 1 if quant else 2
        weights_gb = _estimate_params(vals, num_experts) * bytes_per_param / GiB

    architectures = config.get("architectures") or []
    try:
        return ModelInfo(
            path=path,
            name=Path(path).name if is_local else model.split("/")[-1],
            model_type=model_type,
            architectures=[str(a) for a in architectures]
            if isinstance(architectures, list)
            else [],
            num_layers=vals["num_hidden_layers"],
            hidden_size=vals["hidden_size"],
            intermediate_size=vals["intermediate_size"],
            vocab_size=vals["vocab_size"],
            num_attention_heads=vals["num_attention_heads"],
            num_kv_heads=vals["num_key_value_heads"],
            head_dim=vals["head_dim"],
            max_position=vals["max_position_embeddings"],
            is_moe=num_experts is not None and num_experts > 1,
            num_experts=num_experts,
            is_multimodal=_is_multimodal(cfg),
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
    except ValidationError as e:
        raise ModelInfoError(f"config.json has values AutoDSpark can't use: {e}") from e


def _from_hub(
    repo_id: str, token: str | None
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, float | None]:
    """Read config, tokenizer config, chat template and weight size for a Hub repo id.

    Files already in the Hugging Face cache are read from there; with HF_HUB_OFFLINE set the
    network is never used.
    """
    offline = hub_offline()
    snapshot = _cached_snapshot(repo_id)

    def text(filename: str) -> str | None:
        if snapshot is not None:
            cached = snapshot / filename
            if cached.is_file():
                try:
                    return cached.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as e:
                    raise ModelInfoError(f"Could not read {cached}: {e}") from e
            # huggingface_hub records files it learned don't exist under .no_exist/<rev>/.
            if (snapshot.parent.parent / ".no_exist" / snapshot.name / filename).exists():
                return None
        return None if offline else _hub_file(repo_id, filename, token)

    def as_json(filename: str) -> dict[str, Any] | None:
        raw = text(filename)
        return None if raw is None else _parse_json(raw, f"{filename} of {repo_id}")

    config = as_json("config.json")
    if config is None:
        if offline:
            raise _offline_error(repo_id)
        raise ModelInfoError(f"Could not find config.json for {repo_id} on the Hub.")
    tokenizer_cfg = as_json("tokenizer_config.json")
    chat_template = text("chat_template.jinja")

    weights_gb = _local_weights_gb(snapshot) if snapshot is not None else None
    if weights_gb is None:
        try:
            weights_gb = _index_total_gb(as_json("model.safetensors.index.json"))
        except ModelInfoError:
            weights_gb = None
    if weights_gb is None and not offline:
        weights_gb = _hub_weights_gb(repo_id, token)
    return config, tokenizer_cfg, chat_template, weights_gb
