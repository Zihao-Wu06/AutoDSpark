"""Build the exact command line for each pipeline step.

Pure functions only: they never run anything, so `run --dry-run` and the
tests can show precisely what would execute.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .planner import DataSource, Plan
from .runtime import Runtime


@dataclass
class Cmd:
    argv: list[str]
    gpus: list[int] = field(default_factory=list)
    label: str = ""
    # How CUDA_VISIBLE_DEVICES names a GPU index when that isn't the index itself,
    # e.g. the GPU-... UUID the user restricted us to.
    device_ids: dict[int, str] = field(default_factory=dict)

    @property
    def env(self) -> dict[str, str]:
        if not self.gpus:
            return {}
        # nvidia-smi numbers GPUs in PCI order; make CUDA use the same order. Physical
        # indices (not the user's UUIDs) because vLLM parses CUDA_VISIBLE_DEVICES as
        # integers. The plan only holds GPUs the user allowed, so this never widens access.
        return {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in self.gpus),
        }

    @property
    def venv_bin(self) -> Path | None:
        """The virtualenv `bin/` the command's interpreter lives in, if any.

        Tools inside the env (ninja for FlashInfer's JIT, torchrun, ...) are
        only found when that folder is on PATH, as if the env were activated.
        """
        exe = Path(self.argv[0]) if self.argv else None
        if exe and exe.parent.name == "bin" and (exe.parent.parent / "pyvenv.cfg").exists():
            return exe.parent
        return None

    def shell(self) -> str:
        parts = [f"{k}={shlex.quote(v)}" for k, v in self.env.items()]
        if self.venv_bin:
            parts.append(f"PATH={shlex.quote(str(self.venv_bin))}:$PATH")
        cmd = shlex.join(self.argv)
        return f"{' '.join(parts)} {cmd}" if parts else cmd


def _device_ids(plan: Plan) -> dict[int, str]:
    """GPU index -> the user's own CUDA_VISIBLE_DEVICES name for it, where they differ."""
    return {g.index: g.cuda_id for g in plan.hardware.gpus if g.cuda_id}


def _trust(plan: Plan) -> list[str]:
    # Only models that ship custom modeling code need this.
    return ["--trust-remote-code"] if plan.trust_remote_code else []


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def eval_max_model_len(plan: Plan) -> int:
    want = max(4096, plan.eval.max_tokens + 3072)
    return min(want, plan.model.max_position) if plan.model.max_position else want


def vllm_serve(
    rt: Runtime,
    model: str,
    port: int,
    *,
    tp: int,
    dp: int = 1,
    mem_util: float = 0.9,
    max_model_len: int | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    argv = [
        str(rt.vllm_python),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        str(mem_util),
    ]
    if dp > 1:
        argv += ["--data-parallel-size", str(dp)]
    if max_model_len:
        argv += ["--max-model-len", str(max_model_len)]
    return argv + (extra or [])


# --------------------------------------------------------------------------- stage commands


def regen_server(plan: Plan, rt: Runtime) -> Cmd:
    lay = plan.layout
    return Cmd(
        vllm_serve(
            rt,
            plan.model.path,
            plan.ports.regen,
            tp=lay.regen_tp,
            dp=lay.regen_dp,
            mem_util=lay.regen_mem_util,
            max_model_len=plan.data.regen_max_model_len,
            extra=_trust(plan),
        ),
        gpus=lay.regen_gpus,
        label="vLLM (response regeneration)",
        device_ids=_device_ids(plan),
    )


def regen(
    plan: Plan, rt: Runtime, source: DataSource, outfile: Path, limit: int | None = None
) -> Cmd:
    return Cmd(
        [
            str(rt.speculators_python),
            "-m",
            "speculators",
            "regenerate-responses",
            "--endpoint",
            f"{base_url(plan.ports.regen)}/v1/chat/completions",
            "--model",
            plan.model.path,
            "--dataset",
            source.value,
            "--limit",
            str(limit if limit is not None else source.limit),
            "--max-tokens",
            str(plan.data.regen_max_tokens),
            "--concurrency",
            str(plan.data.regen_concurrency),
            "--outfile",
            str(outfile),
            "--resume",
        ],
        label=f"regenerate responses ({source.slug})",
    )


def prepare(plan: Plan, rt: Runtime, regen_files: list[Path], out_dir: Path) -> Cmd:
    argv = [
        str(rt.speculators_python),
        "-m",
        "speculators",
        "prepare-data",
        "--model",
        plan.model.path,
    ]
    for f in regen_files:
        argv += ["--data", str(f)]
    argv += [
        "--output",
        str(out_dir),
        "--seq-length",
        str(plan.data.seq_len),
        "--overwrite",
    ]
    if plan.draft.draft_vocab_size:
        argv.append("--no-skip-token-freq")  # writes token_freq.pt for the reduced vocab
    return Cmd(argv + _trust(plan), label="prepare data")


def extract_server(plan: Plan, rt: Runtime, hs_path: Path) -> Cmd:
    """vLLM that streams hidden states of the target layers to training."""
    lay = plan.layout
    argv = [
        str(rt.vllm_python),
        str(rt.launch_vllm),
        "train",
        plan.model.path,
        "--target-layer-ids",
        *map(str, plan.draft.target_layer_ids),
        "--hidden-states-path",
        str(hs_path),
        *_trust(plan),
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        str(plan.ports.extract),
        "--tensor-parallel-size",
        str(lay.vllm_tp),
        "--gpu-memory-utilization",
        str(lay.vllm_mem_util),
        "--max-model-len",
        str(plan.data.seq_len + 1),
        *_trust(plan),
    ]
    if lay.vllm_dp > 1:
        argv += ["--data-parallel-size", str(lay.vllm_dp)]
    return Cmd(
        argv,
        gpus=lay.vllm_gpus,
        label="vLLM (hidden-state extraction)",
        device_ids=_device_ids(plan),
    )


def generate_offline(plan: Plan, rt: Runtime, prepared: Path, hs_out: Path) -> Cmd:
    return Cmd(
        [
            str(rt.speculators_python),
            "-m",
            "speculators",
            "generate-offline-data",
            "--preprocessed-data",
            str(prepared),
            "--endpoint",
            f"{base_url(plan.ports.extract)}/v1",
            "--output",
            str(hs_out),
            "--concurrency",
            "32",
            "--validate-outputs",
        ],
        label="cache hidden states",
    )


def train(
    plan: Plan,
    rt: Runtime,
    prepared: Path,
    save_path: Path,
    hs_cache: Path | None = None,
    max_anchors: int | None = None,
) -> Cmd:
    lay, d, t = plan.layout, plan.draft, plan.train
    n = len(lay.train_gpus)
    if n > 1:
        launcher = [
            str(rt.speculators_python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(n),
            "-m",
            "speculators.train",
        ]
    else:
        launcher = [str(rt.speculators_python), "-m", "speculators", "train"]
    argv = launcher + [
        "--verifier-name-or-path",
        plan.model.path,
        "--data-path",
        str(prepared),
        "--save-path",
        str(save_path),
        "--speculator-type",
        "dspark",
        "--epochs",
        str(t.epochs),
        "--lr",
        str(t.lr),
        "--loss-fn",
        json.dumps(t.loss_fn, separators=(",", ":")),
        "--total-seq-len",
        str(plan.data.seq_len),
        "--num-layers",
        str(d.num_layers),
        "--block-size",
        str(d.block_size),
        "--max-anchors",
        str(max_anchors or d.max_anchors),
        "--target-layer-ids",
        *map(str, d.target_layer_ids),
        "--markov-rank",
        str(d.markov_rank),
        "--markov-head-type",
        d.markov_head_type,
        "--enable-confidence-head",
        # The confidence head can only read the Markov head if there is one.
        "--confidence-head-with-markov"
        if d.markov_rank > 0
        else "--no-confidence-head-with-markov",
        "--confidence-head-alpha",
        str(d.confidence_head_alpha),
        "--save-best",
    ]
    if d.draft_vocab_size:
        argv += ["--draft-vocab-size", str(d.draft_vocab_size)]
    if lay.mode == "online":
        argv += [
            "--vllm-endpoint",
            f"{base_url(plan.ports.extract)}/v1",
            "--on-missing",
            "generate",
        ]
    else:
        if hs_cache is None:
            raise ValueError("offline training needs the hidden-state cache path")
        # generate-offline-data skips samples it couldn't produce and still exits 0;
        # "warn" drops those samples at train time instead of crashing every resume.
        argv += ["--hidden-states-path", str(hs_cache), "--on-missing", "warn"]
    return Cmd(
        argv + list(t.extra_args) + _trust(plan),
        gpus=lay.train_gpus,
        label="train DSpark drafter",
        device_ids=_device_ids(plan),
    )


def eval_server(plan: Plan, rt: Runtime, drafter: Path | None) -> Cmd:
    """Baseline server (drafter=None) or the target served with the trained drafter."""
    tp = plan.layout.regen_tp
    gpus = plan.layout.regen_gpus[:tp]
    extra = ["--served-model-name", "target", *_trust(plan)]
    # Always the explicit form: `vllm serve <drafter>` only works when the
    # drafter's verifier is a Hub repo id, and vLLM refuses local verifiers.
    if drafter is not None:
        extra += ["--speculative-config", json.dumps(speculative_config(plan, drafter))]
    return Cmd(
        vllm_serve(
            rt,
            plan.model.path,
            plan.ports.eval,
            tp=tp,
            mem_util=plan.layout.eval_mem_util,
            max_model_len=eval_max_model_len(plan),
            extra=extra,
        ),
        gpus=gpus,
        label="vLLM (eval, with DSpark)" if drafter else "vLLM (eval, baseline)",
        device_ids=_device_ids(plan),
    )


def speculative_config(plan: Plan, drafter: Path) -> dict:
    return {
        "method": "dspark",
        "model": str(drafter),
        "num_speculative_tokens": plan.draft.block_size,
    }


def serve_commands(plan: Plan, drafter: Path) -> list[tuple[str, str]]:
    """User-facing commands to serve the result."""
    spec = json.dumps(speculative_config(plan, drafter))
    tp = plan.layout.regen_tp
    tp_flag = ["--tensor-parallel-size", str(tp)] if tp > 1 else []
    full = ["vllm", "serve", plan.model.path, *tp_flag, "--speculative-config", spec]
    cmds = [("Serve your model with the drafter", shlex.join(full))]
    if not plan.model.is_local:
        # vLLM reads the target from the drafter's config, but only accepts a
        # Hub repo id there, never a local folder.
        short = ["vllm", "serve", str(drafter), *tp_flag]
        cmds.append(("Shorter form (target is a Hub repo id)", shlex.join(short)))
    return cmds
