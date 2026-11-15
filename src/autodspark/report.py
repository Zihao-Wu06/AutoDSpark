"""Write the human-readable result of a run."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from . import commands
from .planner import Plan


def _fmt(x, suffix: str = "") -> str:
    return "-" if x is None else f"{x}{suffix}"


def _headline(speedup: float | None) -> str:
    if not speedup:
        return "Speedup could not be computed."
    if speedup >= 1:
        return f"**{speedup:.2f}x faster** single-stream generation with the DSpark drafter."
    return (
        f"**{speedup:.2f}x the baseline speed**: with this drafter, generation is slower than "
        "without it."
    )


def _advice(plan: Plan, speedup: float | None) -> list[str]:
    notes = []
    if plan.preset == "quick":
        notes.append(
            "This was a `quick` run: it checks that the pipeline works on your machine, with "
            "too little training for a useful drafter. Run `--preset standard` for real gains."
        )
    if speedup is not None and speedup < 1.05:
        small = plan.model.weights_gb < 4
        notes.append(
            "The drafter doesn't speed generation up yet. More training data and epochs raise "
            "acceptance; training on prompts like the ones you serve (`--data`) helps most."
            + (
                " Small target models also gain little from speculative decoding, because "
                "they are already fast relative to the drafter."
                if small
                else ""
            )
        )
    out: list[str] = []
    for n in notes:  # a blank line between quotes keeps them as separate paragraphs
        out += [f"> {n}", ""]
    return out


def _sharing(plan: Plan) -> str:
    lay = plan.layout
    if not lay.shared_gpu:
        return ""
    shared = sorted(set(lay.vllm_gpus) & set(lay.train_gpus)) or lay.train_gpus
    if len(shared) <= 1:
        return " (vLLM and training share one GPU)"
    return f" (each of the {len(shared)} GPUs ran a vLLM replica and a training rank)"


def render(plan: Plan, drafter: Path | None, eval_result: dict | None) -> str:
    m, d, lay = plan.model, plan.draft, plan.layout
    lines = [
        f"# AutoDSpark report: {m.name}",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by AutoDSpark {plan.autodspark_version} "
        f"(Speculators {plan.speculators_version}).",
        "",
    ]
    if eval_result:
        base, spec, cmp = eval_result["baseline"], eval_result["dspark"], eval_result["compare"]
        speedup = cmp.get("speedup")
        lines += [
            "## Result",
            "",
            _headline(speedup),
            "",
            "| | Baseline | With DSpark |",
            "|---|---|---|",
            f"| Output tokens/s (one request at a time) | {base['tokens_per_second']} "
            f"| {spec['tokens_per_second']} |",
            f"| Mean acceptance length (tokens per verify step) | - "
            f"| {_fmt(cmp.get('acceptance_length'))} |",
            f"| Draft token acceptance rate | - | {_fmt(cmp.get('acceptance_rate'))} |",
            "",
            f"Measured on {base['num_requests']} prompts, up to {plan.eval.max_tokens} new tokens "
            f"each, temperature {plan.eval.temperature}.",
            "",
        ]
        lines += _advice(plan, speedup)
        cats = cmp.get("by_category") or {}
        if len(cats) > 1:
            lines += ["| Category | Baseline tok/s | DSpark tok/s | Speedup |", "|---|---|---|---|"]
            for c, v in cats.items():
                b, s = v.get("baseline_tps"), v.get("dspark_tps")
                ratio = f"{s / b:.2f}x" if b and s else "-"
                lines.append(f"| {c} | {_fmt(b)} | {_fmt(s)} | {ratio} |")
            lines.append("")
    if drafter:
        lines += ["## Serve it", ""]
        for title, cmd in commands.serve_commands(plan, drafter):
            lines += [f"{title}:", "", "```bash", cmd, "```", ""]
    lines += [
        "## Training setup",
        "",
        f"- Target model: `{m.path}` ({m.num_layers} layers, hidden {m.hidden_size}, "
        f"vocab {m.vocab_size}, {m.size_label})",
        f"- GPUs: {plan.hardware.num_gpus} x {plan.hardware.min_gpu_gb:g} GB, "
        f"{lay.mode} mode{_sharing(plan)}",
        f"- Data: {plan.data.max_samples:,} samples, sequence length {plan.data.seq_len}, "
        "sources: " + ", ".join(f"{s.value} ({s.limit:,})" for s in plan.data.sources),
        f"- Drafter: {d.num_layers} layers, block size {d.block_size}, target layers "
        f"{d.target_layer_ids}, draft vocab {d.draft_vocab_size or 'full'}",
        f"- Training: {plan.train.epochs} epochs, lr {plan.train.lr}, preset `{plan.preset}`",
        "",
    ]
    if plan.warnings:
        lines += ["## Notes", ""] + [f"- {w}" for w in plan.warnings] + [""]
    return "\n".join(lines)
