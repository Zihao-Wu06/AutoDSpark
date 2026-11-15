"""Turn (model, hardware, preset) into a concrete, editable training plan.

Every number here is a starting point, not a law. The plan is written to
plan.yaml so users can read and change it before `autodspark run`.
Memory figures are rough estimates; the runner retries with smaller
settings if training still runs out of memory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from . import SPECULATORS_VERSION, __version__
from .compat import known_issues
from .hardware import GPU, Hardware
from .model_info import GiB, ModelInfo

DEFAULT_PUBLIC_DATASET = "open-perfectblend"
PUBLIC_DATASETS = ("open-perfectblend", "ultrachat", "sharegpt", "magpie", "nemotron", "gsm8k")

PRESETS: dict[str, dict[str, int]] = {
    # Checks the whole pipeline end to end in minutes. Acceptance will be low.
    "quick": {"samples": 1000, "epochs": 1, "seq_len": 2048},
    # A first real drafter in a few GPU-hours on one large GPU (e.g. a 4B model
    # on one 80 GB card in about six hours).
    "medium": {"samples": 10000, "epochs": 3, "seq_len": 2048},
    # A useful drafter for most chat/code workloads.
    "standard": {"samples": 20000, "epochs": 5, "seq_len": 4096},
    # Closer to published recipes; needs much more GPU time.
    "full": {"samples": 100000, "epochs": 5, "seq_len": 8192},
}

GPU_HEADROOM = 0.92  # never plan to fill more than this share of a GPU
SMALL_GPU_RATIO = 0.5  # GPUs with less usable memory than this share of the biggest are left out
REGEN_MEM_UTIL = 0.9  # vLLM --gpu-memory-utilization when it has the GPUs to itself
EVAL_MEM_UTIL = 0.85
MIN_ANCHORS = 512
MIN_SEQ_LEN = 2048
MIN_DRAFT_LAYERS = 3


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


class DataSource(BaseModel):
    kind: Literal["preset", "file"]
    value: str  # preset name or file path
    limit: int  # number of conversations to regenerate
    slug: str = ""  # output file name; unique within a plan

    @model_validator(mode="after")
    def _default_slug(self) -> DataSource:
        if not self.slug:
            self.slug = _slugify(Path(self.value).stem if self.kind == "file" else self.value)
        return self


class DataPlan(BaseModel):
    sources: list[DataSource]
    max_samples: int  # conversations in total (multi-turn ones give several rows)
    seq_len: int
    regen_max_tokens: int
    regen_max_model_len: int
    regen_concurrency: int = 64


class DraftPlan(BaseModel):
    num_layers: int
    block_size: int = 8
    max_anchors: int
    draft_vocab_size: int | None
    target_layer_ids: list[int]
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    confidence_head_alpha: float = 1.0


class TrainPlan(BaseModel):
    epochs: int
    lr: float = 3e-4
    loss_fn: dict[str, float] = Field(default_factory=lambda: {"ce": 0.1, "tv": 0.9})
    # Passed through to `speculators train` as-is, e.g. ["--gradient-checkpointing",
    # "--num-workers", "2"] to save GPU memory when vLLM shares the GPUs.
    extra_args: list[str] = Field(default_factory=list)


class Layout(BaseModel):
    """Which GPUs do what. Indices are physical GPU ids."""

    mode: Literal["online", "offline"]
    shared_gpu: bool = False  # vLLM and training share one GPU (online, 1 GPU)
    regen_gpus: list[int]
    regen_tp: int
    vllm_gpus: list[int]
    vllm_tp: int
    vllm_mem_util: float
    train_gpus: list[int]
    # Lowered below the defaults only when other processes already hold GPU memory.
    regen_mem_util: float = REGEN_MEM_UTIL
    eval_mem_util: float = EVAL_MEM_UTIL

    @property
    def regen_dp(self) -> int:
        return max(1, len(self.regen_gpus) // self.regen_tp)

    @property
    def vllm_dp(self) -> int:
        return max(1, len(self.vllm_gpus) // self.vllm_tp)


class EvalPlan(BaseModel):
    prompts_file: str | None = None  # None = built-in prompt set
    num_prompts: int = 40
    max_tokens: int = 512
    temperature: float = 0.0


class Ports(BaseModel):
    regen: int = 8100
    extract: int = 8101
    eval: int = 8102


class Estimates(BaseModel):
    weights_gb: float
    vllm_gb: float
    train_gb_per_gpu: float
    offline_disk_gb: float | None = None
    train_tokens_m: float


class Plan(BaseModel):
    autodspark_version: str = __version__
    speculators_version: str = SPECULATORS_VERSION
    preset: str
    model: ModelInfo
    hardware: Hardware
    data: DataPlan
    draft: DraftPlan
    train: TrainPlan
    layout: Layout
    eval: EvalPlan = Field(default_factory=EvalPlan)
    ports: Ports = Field(default_factory=Ports)
    trust_remote_code: bool = False
    estimates: Estimates
    warnings: list[str] = []
    errors: list[str] = []

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        header = (
            "# AutoDSpark training plan. Edit any value, then run:\n"
            f"#   autodspark run --plan {path.name}\n"
        )
        path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    @classmethod
    def load(cls, path: Path) -> Plan:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def problems(self) -> list[str]:
        """Checks for values that would break a run, e.g. after editing plan.yaml."""
        out: list[str] = []
        m, d, data = self.model, self.draft, self.data
        ids = d.target_layer_ids
        if not ids or len(set(ids)) != len(ids) or not all(0 <= i < m.num_layers for i in ids):
            out.append(
                f"draft.target_layer_ids must be distinct and in [0, {m.num_layers - 1}], got {ids}"
            )
        if m.max_position and data.seq_len + 1 > m.max_position:
            out.append(f"data.seq_len must be below the model's context ({m.max_position}).")
        if m.max_position and data.regen_max_model_len > m.max_position:
            out.append(f"data.regen_max_model_len can't exceed {m.max_position}.")
        if data.regen_max_tokens >= data.regen_max_model_len:
            out.append("data.regen_max_tokens must be smaller than data.regen_max_model_len.")
        if d.num_layers < 1 or d.block_size < 2 or d.max_anchors < 1 or d.markov_rank < 0:
            out.append("draft settings must be positive (block_size at least 2).")
        known = {g.index for g in self.hardware.gpus}
        used = set(self.layout.regen_gpus + self.layout.vllm_gpus + self.layout.train_gpus)
        if known and not used <= known:
            out.append(f"layout uses GPUs {sorted(used - known)} that aren't in hardware.gpus.")
        slugs = [s.slug for s in data.sources]
        if len(set(slugs)) != len(slugs):
            out.append("data.sources need unique slugs.")
        return out


class PlanError(RuntimeError):
    pass


# --------------------------------------------------------------------------- GPUs


def mem_util_cap(gpu: GPU) -> float:
    """Highest --gpu-memory-utilization vLLM can start with, given what others hold.

    vLLM refuses to start unless util x total memory is free. Idle GPUs allow
    more than the defaults, so this only bites on busy ones.
    """
    if gpu.memory_gb <= 0:
        return 0.0
    share = round((gpu.free_gb / gpu.memory_gb - 0.03) * 100, 6)
    return max(0.0, math.floor(share) / 100)


def usable_gb(gpu: GPU) -> float:
    """Memory one GPU offers a plan: the headroom-limited total, or less if it is busy."""
    return min(gpu.memory_gb * GPU_HEADROOM, mem_util_cap(gpu) * gpu.memory_gb)


def gpu_budget(hw: Hardware) -> float:
    return min((usable_gb(g) for g in hw.gpus), default=0.0)


def is_busy(gpu: GPU) -> bool:
    return gpu.memory_used_gb > max(1.0, 0.05 * gpu.memory_gb)


def capped_util(util: float, hw: Hardware, gpus: list[int]) -> float:
    """`util`, lowered if needed so vLLM can start on every one of `gpus`."""
    caps = [mem_util_cap(g) for g in hw.gpus if g.index in gpus]
    return round(min([util, *caps]), 2)


def _describe(gpu: GPU) -> str:
    busy = f", {gpu.memory_used_gb:g} GB in use" if is_busy(gpu) else ""
    return f"GPU {gpu.index} ({gpu.name}, {gpu.memory_gb:g} GB{busy})"


def select_gpus(hw: Hardware) -> tuple[Hardware, list[str]]:
    """Drop GPUs far smaller (or busier) than the best one, and warn about busy ones.

    Every GPU in a plan is sized like the smallest one, so a small display card
    next to a big GPU would otherwise shrink the whole plan to fit the small card.
    """
    notes: list[str] = []
    best = max((usable_gb(g) for g in hw.gpus), default=0.0)
    keep = [g for g in hw.gpus if usable_gb(g) >= SMALL_GPU_RATIO * best]
    dropped = [g for g in hw.gpus if g not in keep]
    if dropped:
        ids = ",".join(g.visible_id for g in keep)
        notes.append(
            f"Ignoring {', '.join(_describe(g) for g in dropped)}: much less memory is usable "
            f"there than on the others. Planning with GPU {', '.join(str(g.index) for g in keep)}"
            f". Set CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES={ids}) to pick GPUs yourself."
        )
        hw = hw.model_copy(update={"gpus": keep})
    for g in keep:
        if is_busy(g):
            notes.append(
                f"{_describe(g)} is partly used by another process. The plan only counts the "
                f"{g.free_gb:.0f} GB that is free and, if needed, lowers vLLM's "
                "--gpu-memory-utilization to fit; free the GPU (or exclude it with "
                "CUDA_VISIBLE_DEVICES) for a faster run."
            )
    return hw, notes


# --------------------------------------------------------------------------- estimates


def kv_bytes_per_token(model: ModelInfo) -> float | None:
    heads = model.num_attention_heads
    if not heads or model.uses_mla:
        return None
    head_dim = model.head_dim or model.hidden_size / heads
    kv_heads = model.num_kv_heads or heads
    return 2 * model.num_layers * kv_heads * head_dim * 2  # K and V, bf16


def vllm_need_gb(model: ModelInfo, seq_len: int) -> float:
    """Memory to serve the target: weights + a small KV cache + runtime overhead."""
    per_tok = kv_bytes_per_token(model)
    kv_tokens = max(4 * seq_len, 16384)
    kv_gb = per_tok * kv_tokens / GiB if per_tok else max(2.0, 0.15 * model.weights_gb)
    return model.weights_gb * 1.05 + kv_gb + 1.5


def train_need_gb(model: ModelInfo, draft: DraftPlan, seq_len: int) -> float:
    """Very rough per-GPU memory for DSpark training (data parallel, no FSDP)."""
    h = model.hidden_size
    inter = model.intermediate_size or 4 * h
    vocab = model.vocab_size
    vd = draft.draft_vocab_size or vocab
    k = len(draft.target_layer_ids)
    per_layer = 4 * h * h + 3 * h * inter
    trainable = (
        draft.num_layers * per_layer
        + k * h * h  # fc that fuses target hidden states
        + vd * h  # draft lm_head
        + draft.markov_rank * (vd + h)
    )
    frozen = 2 * vocab * h  # embeddings + verifier lm_head
    param_bytes = trainable * 14 + frozen * 2
    q_tokens = draft.max_anchors * draft.block_size
    act_bytes = (
        draft.num_layers * q_tokens * h * 34
        + seq_len * h * (k + 1) * 2
        + q_tokens * vd * 6
        + q_tokens * vocab * 2
    )
    return (param_bytes + act_bytes) * 1.2 / GiB + 2.0


def offline_disk_gb(model: ModelInfo, draft: DraftPlan, seq_len: int, samples: int) -> float:
    avg_len = 0.6 * seq_len  # regenerated answers rarely fill the whole window
    per_sample = avg_len * (len(draft.target_layer_ids) + 1) * model.hidden_size * 2
    return samples * per_sample / GiB


# --------------------------------------------------------------------------- choices


def default_target_layers(num_layers: int) -> list[int]:
    """Same default Speculators uses: early, middle and late layers."""
    if num_layers >= 8:
        return [2, num_layers // 2, num_layers - 3]
    picks = sorted({max(1, round(num_layers * f)) for f in (0.25, 0.5, 0.75)})
    return [p for p in picks if p < num_layers] or [max(1, num_layers - 1)]


def default_draft_layers(model: ModelInfo) -> int:
    approx_params_b = model.weights_gb * GiB / 2 / 1e9  # bf16-equivalent
    return 3 if approx_params_b < 3 else 5


def _round_down(x: float, step: int) -> int:
    return int(x // step * step)


def count_conversations(path: Path) -> int:
    """Conversations in a JSONL file, or in a JSON file holding one array of them.

    `speculators regenerate-responses` reads local files with the Hugging Face
    "json" loader, which takes both forms.
    """
    with path.open("rb") as f:
        head = f.read(4096).removeprefix(b"\xef\xbb\xbf").lstrip()
    if head.startswith(b"["):
        try:
            data = json.loads(path.read_bytes().decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise PlanError(
                f"{path.name}: looks like a JSON array but isn't valid JSON ({e})."
            ) from e
        return len(data)
    rows, first = 0, None
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                rows += 1
                first = first if first is not None else line
    if first is None:
        return 0
    try:
        ok = isinstance(json.loads(first.decode("utf-8-sig")), dict)
    except (json.JSONDecodeError, UnicodeDecodeError):
        ok = False
    if not ok:
        raise PlanError(
            f"{path.name}: expected one JSON object per line (JSONL) or a JSON array of "
            "conversations, but the first line isn't a JSON object. Pretty-printed objects "
            "aren't supported; write one conversation per line."
        )
    return rows


def build_sources(
    samples: int, user_files: list[str], public: str | None
) -> tuple[list[DataSource], list[str]]:
    notes: list[str] = []
    sources: list[DataSource] = []
    remaining = samples
    for f in user_files:
        p = Path(f).expanduser()
        if not p.is_file():
            raise PlanError(f"Data file not found: {f}")
        if p.suffix.lower() not in (".jsonl", ".json"):
            raise PlanError(f"{p.name}: data files must be .jsonl or .json")
        rows = count_conversations(p)
        if rows == 0:
            notes.append(f"Skipping {p.name}: the file is empty (no conversations in it).")
            continue
        take = min(rows, remaining)
        if take <= 0:
            notes.append(f"Skipping {p.name}: the sample budget is already used up.")
            continue
        sources.append(DataSource(kind="file", value=str(p.resolve()), limit=take))
        remaining -= take
    if public and public != "none" and remaining > 0:
        if public not in PUBLIC_DATASETS:
            raise PlanError(f"Unknown public dataset {public!r}. Choose from {PUBLIC_DATASETS}.")
        sources.append(DataSource(kind="preset", value=public, limit=remaining))
    if not sources:
        raise PlanError("No training data: pass --data and/or keep a --public-data preset.")
    used: set[str] = set()
    for src in sources:  # two files named data.jsonl must not share one output file
        base, n = src.slug, 2
        while src.slug in used:
            src.slug = f"{base}_{n}"
            n += 1
        used.add(src.slug)
    if user_files and public and public != "none" and remaining > 0:
        notes.append(
            f"Your data gives {samples - remaining} samples; topping up with {remaining} "
            f"from {public}."
        )
    return sources, notes


def _layout(
    model: ModelInfo,
    hw: Hardware,
    draft: DraftPlan,
    seq_len: int,
) -> tuple[Layout, float, float]:
    n, gpu_gb = hw.num_gpus, hw.min_gpu_gb
    v_need = vllm_need_gb(model, seq_len)
    t_need = train_need_gb(model, draft, seq_len)
    budget = gpu_budget(hw)

    tp = next((t for t in (1, 2, 4, 8) if t <= n and v_need / t <= budget), None)
    if tp is None:
        busy = any(is_busy(g) for g in hw.gpus)
        raise PlanError(
            f"{model.name} needs about {v_need:.0f} GB to serve, more than "
            f"{n} x {gpu_gb:.0f} GB GPUs can hold on one machine"
            + (
                f" (about {budget:.0f} GB usable on each; other processes hold the rest)"
                if busy
                else ""
            )
            + ". Use a smaller or quantized model, or more GPUs."
        )
    all_gpus = [g.index for g in hw.gpus]
    regen_count = n // tp * tp

    def util(need_per_gpu: float) -> float:
        return round(min(0.9, max(0.1, math.ceil(need_per_gpu / gpu_gb * 100) / 100 + 0.02)), 2)

    if n == 1:
        if v_need + t_need <= budget:
            layout = Layout(
                mode="online",
                shared_gpu=True,
                regen_gpus=all_gpus,
                regen_tp=1,
                vllm_gpus=all_gpus,
                vllm_tp=1,
                vllm_mem_util=util(v_need),
                train_gpus=all_gpus,
            )
        else:
            layout = Layout(
                mode="offline",
                regen_gpus=all_gpus,
                regen_tp=1,
                vllm_gpus=all_gpus,
                vllm_tp=1,
                vllm_mem_util=0.9,
                train_gpus=all_gpus,
            )
        return _cap_utils(layout, hw), v_need, t_need

    if n - tp >= 1:
        # Give vLLM about half the GPUs (in whole TP groups), training the rest.
        vllm_count = max(tp, (n // 2) // tp * tp)
        if n - vllm_count < 1:
            vllm_count = tp
        layout = Layout(
            mode="online",
            regen_gpus=all_gpus[:regen_count],
            regen_tp=tp,
            vllm_gpus=all_gpus[:vllm_count],
            vllm_tp=tp,
            vllm_mem_util=0.9,
            train_gpus=all_gpus[vllm_count:],
        )
    else:
        layout = Layout(
            mode="offline",
            regen_gpus=all_gpus[:regen_count],
            regen_tp=tp,
            vllm_gpus=all_gpus[:regen_count],
            vllm_tp=tp,
            vllm_mem_util=0.9,
            train_gpus=all_gpus,
        )
    return _cap_utils(layout, hw), v_need, t_need


def _cap_utils(layout: Layout, hw: Hardware) -> Layout:
    """Lower vLLM memory shares where other processes already hold GPU memory."""
    return layout.model_copy(
        update={
            "vllm_mem_util": capped_util(layout.vllm_mem_util, hw, layout.vllm_gpus),
            "regen_mem_util": capped_util(REGEN_MEM_UTIL, hw, layout.regen_gpus),
            "eval_mem_util": capped_util(EVAL_MEM_UTIL, hw, layout.regen_gpus[: layout.regen_tp]),
        }
    )


def make_plan(
    model: ModelInfo,
    hw: Hardware,
    preset: str = "standard",
    *,
    data_files: list[str] | None = None,
    public_data: str | None = DEFAULT_PUBLIC_DATASET,
    samples: int | None = None,
    epochs: int | None = None,
    seq_len: int | None = None,
    num_draft_layers: int | None = None,
    block_size: int | None = None,
    target_layer_ids: list[int] | None = None,
    mode: str | None = None,
    eval_prompts: str | None = None,
    trust_remote_code: bool | None = None,
) -> Plan:
    if preset not in PRESETS:
        raise PlanError(f"Unknown preset {preset!r}. Choose from {list(PRESETS)}.")
    p = PRESETS[preset]
    warnings: list[str] = known_issues(model)
    errors: list[str] = []
    if trust_remote_code is None:
        trust_remote_code = model.needs_remote_code
    if model.needs_remote_code and trust_remote_code:
        warnings.append(
            "This model ships its own modeling code, so --trust-remote-code is passed to "
            "vLLM and Speculators. Only train against models whose code you trust."
        )
    elif model.needs_remote_code:
        errors.append("This model needs --trust-remote-code to load.")

    samples = samples or p["samples"]
    epochs = epochs or p["epochs"]
    seq = seq_len or p["seq_len"]
    # The extraction server runs with max_model_len = seq + 1, which must fit
    # in the model's context window.
    seq_cap = model.max_position - 1 if model.max_position else None
    min_seq = min(MIN_SEQ_LEN, seq_cap) if seq_cap else MIN_SEQ_LEN
    if seq_cap and seq > seq_cap:
        seq = seq_cap
        warnings.append(f"Sequence length capped at {seq} by the model's context window.")

    layers = target_layer_ids or default_target_layers(model.num_layers)
    # The final layer is added by Speculators itself, so ids stop below num_layers.
    bad = [i for i in layers if not 0 <= i < model.num_layers]
    if bad or len(set(layers)) != len(layers):
        raise PlanError(
            f"target_layer_ids must be distinct and within [0, {model.num_layers - 1}], "
            f"got {layers}"
        )

    draft = DraftPlan(
        num_layers=num_draft_layers or default_draft_layers(model),
        block_size=block_size or 8,
        max_anchors=max(MIN_ANCHORS, _round_down(0.75 * seq, 256)),
        draft_vocab_size=32000 if model.vocab_size > 64000 else None,
        target_layer_ids=layers,
    )

    if hw.num_gpus == 0:
        if hw.gpu_problem:
            raise PlanError(hw.gpu_problem)
        raise PlanError(
            "No NVIDIA GPU found. Run this on a machine with NVIDIA GPUs, or preview a plan "
            "with --gpus, e.g. `autodspark plan --model ... --gpus 1x80`."
        )
    hw, gpu_notes = select_gpus(hw)
    warnings.extend(gpu_notes)

    # Shrink anchors, then sequence length, until training fits in one GPU.
    budget = gpu_budget(hw)
    seq_before_fit = seq
    layers_before_fit = draft.num_layers
    while train_need_gb(model, draft, seq) > budget:
        if draft.max_anchors > MIN_ANCHORS:
            draft.max_anchors = max(MIN_ANCHORS, draft.max_anchors // 2)
        elif seq > min_seq:
            seq = max(min_seq, seq // 2)
        elif draft.num_layers > MIN_DRAFT_LAYERS and num_draft_layers is None:
            draft.num_layers -= 1
        else:
            errors.append(
                f"Training a {draft.num_layers}-layer drafter for this model needs about "
                f"{train_need_gb(model, draft, seq):.0f} GB per GPU even at the smallest "
                f"settings; your GPUs have {hw.min_gpu_gb:.0f} GB, of which about "
                f"{budget:.0f} GB is usable."
            )
            break
    if seq != seq_before_fit and not errors:
        warnings.append(f"Sequence length lowered to {seq} so training fits in GPU memory.")
    if draft.num_layers != layers_before_fit and not errors:
        warnings.append(
            f"Drafter depth lowered from {layers_before_fit} to {draft.num_layers} layers so "
            "training fits in GPU memory; expect somewhat lower acceptance."
        )

    # On a single GPU, prefer sharing it with vLLM (online) over caching hidden
    # states on disk (offline), even if that means fewer anchors per step.
    if hw.num_gpus == 1:
        anchors_before = draft.max_anchors

        def shared_fits() -> bool:
            return vllm_need_gb(model, seq) + train_need_gb(model, draft, seq) <= budget

        while not shared_fits() and draft.max_anchors > MIN_ANCHORS:
            draft.max_anchors = max(MIN_ANCHORS, draft.max_anchors // 2)
        if not shared_fits():
            draft.max_anchors = anchors_before  # going offline; training gets the whole GPU

    layout, v_need, t_need = _layout(model, hw, draft, seq)
    if mode:
        if mode not in ("online", "offline"):
            raise PlanError("--mode must be online or offline")
        if mode == "online" and layout.mode == "offline":
            errors.append("Online mode doesn't fit on these GPUs; use the planned offline mode.")
        elif mode == "offline" and layout.mode == "online":
            layout = layout.model_copy(
                update={
                    "mode": "offline",
                    "shared_gpu": False,
                    "vllm_gpus": layout.regen_gpus,
                    "vllm_tp": layout.regen_tp,
                    "vllm_mem_util": capped_util(REGEN_MEM_UTIL, hw, layout.regen_gpus),
                    "train_gpus": [g.index for g in hw.gpus],
                }
            )

    sources, notes = build_sources(samples, data_files or [], public_data)
    warnings.extend(notes)

    disk = None
    if layout.mode == "offline":
        disk = offline_disk_gb(model, draft, seq, samples)
        warnings.append(
            f"Offline mode caches hidden states on disk: about {disk:,.0f} GB for "
            f"{samples:,} samples. Online mode would avoid this but needs more GPU memory."
        )
        if disk > hw.disk_free_gb * 0.9:
            errors.append(
                f"Not enough disk: offline mode needs about {disk:,.0f} GB, "
                f"{hw.disk_free_gb:,.0f} GB is free. Lower --samples or free up space."
            )
    if hw.workdir_on_network_fs:
        warnings.append(
            f"The work folder is on a network filesystem ({hw.workdir_fs}). Hidden states "
            "move through it, so training will be much slower. Use a local disk if you can."
        )

    regen_conc = min(256, 64 * max(1, len(layout.regen_gpus) // layout.regen_tp))
    # Prompt + answer must fit the regen server's context; vLLM rejects requests
    # whose prompt length + max_tokens exceed it.
    regen_ctx = min(2 * seq, model.max_position) if model.max_position else 2 * seq
    regen_tokens = min(seq, regen_ctx // 2)
    return Plan(
        preset=preset,
        model=model,
        hardware=hw,
        data=DataPlan(
            sources=sources,
            max_samples=samples,
            seq_len=seq,
            regen_max_tokens=regen_tokens,
            regen_max_model_len=regen_ctx,
            regen_concurrency=regen_conc,
        ),
        draft=draft,
        train=TrainPlan(epochs=epochs),
        layout=layout,
        eval=EvalPlan(prompts_file=str(Path(eval_prompts).resolve()) if eval_prompts else None),
        estimates=Estimates(
            weights_gb=model.weights_gb,
            vllm_gb=round(v_need, 1),
            train_gb_per_gpu=round(t_need, 1),
            offline_disk_gb=round(disk, 1) if disk is not None else None,
            train_tokens_m=round(samples * 0.6 * seq / 1e6, 1),
        ),
        warnings=warnings,
        errors=errors,
        trust_remote_code=trust_remote_code,
    )
