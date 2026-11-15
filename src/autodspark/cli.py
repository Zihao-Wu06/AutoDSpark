"""`autodspark` command line."""

from __future__ import annotations

import contextlib
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import SPECULATORS_VERSION, __version__, commands, hardware
from .model_info import ModelInfoError, inspect_model
from .pipeline import (
    STAGES,
    Paths,
    Pipeline,
    State,
    WorkdirBusy,
    WorkdirLock,
    find_drafter,
    stages_for,
)
from .planner import DEFAULT_PUBLIC_DATASET, PRESETS, Plan, PlanError, make_plan
from .procs import StepFailed, install_signal_handlers
from .runtime import (
    Runtime,
    fetch_speculators_src,
    flashinfer_status,
    missing_setup_tools,
    setup_commands,
    speculators_archive_url,
)

app = typer.Typer(
    name="autodspark",
    help="Train a DSpark speculative-decoding drafter for your own model, in one command.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version(value: bool) -> None:
    if value:
        console.print(f"autodspark {__version__} (Speculators {SPECULATORS_VERSION})")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    pass


# --------------------------------------------------------------------------- shared options

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Local HF model folder or Hub id.")]
PresetOpt = Annotated[str, typer.Option(help=f"How much training to do: {', '.join(PRESETS)}.")]
DataOpt = Annotated[
    list[str] | None,
    typer.Option("--data", help="Your own JSONL prompts/conversations (repeatable)."),
]
PublicOpt = Annotated[
    str,
    typer.Option(help="Public dataset to fill the sample budget, or 'none'."),
]
WorkdirOpt = Annotated[
    Path | None, typer.Option(help="Folder for this run (default: runs/<model>-<preset>).")
]
GpusOpt = Annotated[
    str | None,
    typer.Option(
        help="Pretend hardware for planning, e.g. 1x24 or 4x80. Only for plan / --dry-run."
    ),
]
HomeOpt = Annotated[Path | None, typer.Option(help="Where AutoDSpark keeps its environments.")]


def _runtime(home: Path | None) -> Runtime:
    return Runtime.resolve(home=home)


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _require_linux(what: str) -> None:
    if not _is_linux():
        console.print(
            f"[red]{what} needs Linux with NVIDIA GPUs[/] (this is {sys.platform}). On this "
            "machine you can still preview: `autodspark plan --gpus 1x80 ...` or "
            "`autodspark run --dry-run --gpus 1x80 ...`."
        )
        raise typer.Exit(2)


def _resume_hint(workdir: Path) -> str:
    return f"autodspark run --workdir {shlex.quote(str(workdir))}"


def _say(message: str) -> None:
    """Print, unless the terminal is gone (e.g. after an ssh drop)."""
    with contextlib.suppress(OSError, ValueError):
        console.print(message)


def _model_ref(model: str) -> str:
    """`--model` as the planner records it: a resolved local folder, else the Hub id."""
    local = Path(model).expanduser()
    return str(local.resolve()) if local.is_dir() else model


def _default_workdir(model: str, preset: str) -> Path:
    name = Path(model).name if Path(model).expanduser().exists() else model.split("/")[-1]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return Path("runs") / f"{safe}-{preset}"


def _build_plan(
    model: str,
    preset: str,
    data: list[str] | None,
    public_data: str,
    workdir: Path,
    gpus: str | None,
    samples: int | None,
    epochs: int | None,
    seq_len: int | None,
    draft_layers: int | None,
    eval_prompts: str | None,
    trust_remote_code: bool | None,
) -> Plan:
    try:
        info = inspect_model(model)
    except ModelInfoError as e:
        console.print(f"[red]Model problem:[/] {e}")
        raise typer.Exit(2) from e
    try:
        hw = hardware.simulated(gpus) if gpus else hardware.detect(workdir)
        return make_plan(
            info,
            hw,
            preset,
            data_files=data,
            public_data=public_data,
            samples=samples,
            epochs=epochs,
            seq_len=seq_len,
            num_draft_layers=draft_layers,
            eval_prompts=eval_prompts,
            trust_remote_code=trust_remote_code,
        )
    except (PlanError, ValueError) as e:
        console.print(f"[red]Can't plan this run:[/] {e}")
        raise typer.Exit(2) from e


def print_plan(plan: Plan) -> None:
    m, lay, d, data = plan.model, plan.layout, plan.draft, plan.data
    t = Table(title=f"Plan for {m.name}", show_header=False, title_justify="left")
    t.add_column(style="bold")
    t.add_column()
    flags = [
        x
        for x, on in (
            ("MoE", m.is_moe),
            ("multimodal", m.is_multimodal),
            ("reasoning", m.is_reasoning),
            (m.quant_method, m.quant_method),
        )
        if on
    ]
    t.add_row(
        "Model",
        f"{m.model_type}, {m.num_layers} layers, hidden {m.hidden_size}, vocab {m.vocab_size}, "
        f"{m.size_label}" + (f" ({', '.join(flags)})" if flags else ""),
    )
    gpu_names = ", ".join(sorted({g.name for g in plan.hardware.gpus})) or "none"
    t.add_row("GPUs", f"{plan.hardware.num_gpus} x {plan.hardware.min_gpu_gb:g} GB ({gpu_names})")
    if lay.mode == "online" and lay.shared_gpu:
        how = (
            f"online, vLLM and training share GPU {lay.vllm_gpus} ({lay.vllm_mem_util:.0%} to vLLM)"
        )
    elif lay.mode == "online":
        how = (
            f"online: vLLM on GPUs {lay.vllm_gpus} (TP {lay.vllm_tp}), training on {lay.train_gpus}"
        )
    else:
        how = "offline: cache hidden states on disk, then train on all GPUs"
    t.add_row("Mode", how)
    t.add_row(
        "Data",
        f"{data.max_samples:,} samples, seq len {data.seq_len}: "
        + ", ".join(
            f"{s.value if s.kind == 'preset' else Path(s.value).name} ({s.limit:,})"
            for s in data.sources
        ),
    )
    t.add_row(
        "Drafter",
        f"DSpark, {d.num_layers} layers, block {d.block_size}, target layers "
        f"{d.target_layer_ids}, draft vocab {d.draft_vocab_size or 'full'}, "
        f"max anchors {d.max_anchors}",
    )
    t.add_row("Training", f"{plan.train.epochs} epochs, lr {plan.train.lr}, preset {plan.preset}")
    est = plan.estimates
    mem = f"vLLM ~{est.vllm_gb} GB, training ~{est.train_gb_per_gpu} GB per GPU (rough)"
    if est.offline_disk_gb is not None:
        mem += f", disk ~{est.offline_disk_gb:,.0f} GB"
    t.add_row("Estimates", mem)
    console.print(t)
    for w in plan.warnings:
        console.print(f"[yellow]note:[/] {w}")
    for e in plan.errors:
        console.print(f"[red]blocker:[/] {e}")


def _fail(e: StepFailed, workdir: Path) -> None:
    console.print(f"\n[red bold]Failed:[/] {e}")
    if e.tail:
        console.print("[dim]Last lines of the log:[/]")
        console.print(e.tail, markup=False, highlight=False)
    if e.log:
        console.print(f"Full log: {e.log}")
    console.print(f"Fix the problem, then resume with: {_resume_hint(workdir)}")


# --------------------------------------------------------------------------- commands


@app.command()
def setup(
    home: HomeOpt = None,
    vllm: Annotated[str | None, typer.Option(help="vLLM pip spec, e.g. 'vllm==0.33.0'.")] = None,
    python: Annotated[str, typer.Option(help="Python version for both envs.")] = "3.12",
    dry_run: Annotated[bool, typer.Option(help="Only print the commands.")] = False,
) -> None:
    """Create the Speculators and vLLM environments (needs uv)."""
    if not dry_run:
        _require_linux("Setup")
    rt = _runtime(home)
    cmds = setup_commands(rt, vllm, python)
    missing = missing_setup_tools()
    if missing and not dry_run:
        console.print(
            f"[red]Missing tools:[/] {', '.join(missing)}. Install them first "
            "(https://docs.astral.sh/uv/, or `pip install uv`)."
        )
        raise typer.Exit(2)
    for c in cmds:
        console.print(f"[bold cyan]$[/] {' '.join(c)}", soft_wrap=True)
        if not dry_run and subprocess.run(c).returncode != 0:
            console.print("[red]Setup step failed.[/]")
            raise typer.Exit(1)
    if not rt.launch_vllm.exists():
        console.print(
            f"[bold cyan]$[/] download {speculators_archive_url()} -> {rt.speculators_src}"
        )
        if not dry_run:
            try:
                fetch_speculators_src(rt)
            except Exception as e:  # noqa: BLE001
                console.print(
                    f"[red]Could not download the Speculators scripts:[/] {e}\n"
                    "Set HTTPS_PROXY if GitHub is blocked, or unpack the source yourself and "
                    "point AUTODSPARK_SPECULATORS_SRC at it."
                )
                raise typer.Exit(1) from e
    if not dry_run:
        console.print(f"[green]Done.[/] Environments are in {rt.home}. Next: autodspark doctor")


@app.command()
def doctor(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Also check this model.")
    ] = None,
    workdir: Annotated[Path, typer.Option(help="Where runs will be written.")] = Path("runs"),
    home: HomeOpt = None,
) -> None:
    """Check GPUs, environments, disk and (optionally) a model before training."""
    ok = True
    t = Table(show_header=False, title="AutoDSpark doctor", title_justify="left")
    t.add_column(width=3)
    t.add_column(style="bold")
    t.add_column()

    def row(good: bool, name: str, detail: str, warn_only: bool = False) -> None:
        nonlocal ok
        mark = "[green]✓[/]" if good else ("[yellow]![/]" if warn_only else "[red]✗[/]")
        if not good and not warn_only:
            ok = False
        t.add_row(mark, name, detail)

    hw = hardware.detect(workdir)
    row(
        hw.num_gpus > 0,
        "NVIDIA GPUs",
        ", ".join(f"{g.index}: {g.name} {g.memory_gb:g} GB" for g in hw.gpus)
        or hw.gpu_problem
        or "none found (nvidia-smi missing or no GPUs)",
    )
    if hw.driver_version:
        row(True, "Driver", hw.driver_version)
    row(
        hw.disk_free_gb > 100,
        "Disk free",
        f"{hw.disk_free_gb:,.0f} GB in {workdir.resolve()}",
        warn_only=True,
    )
    if hw.workdir_on_network_fs:
        row(
            False,
            "Work folder",
            f"on a network filesystem ({hw.workdir_fs}); training will be "
            "slow. Use --workdir on a local disk.",
            warn_only=True,
        )
    rt = _runtime(home)
    for name, good, detail in rt.check():
        # A passing check can still carry a warning (e.g. an untested vLLM version).
        warn = good and "warning:" in detail
        row(good and not warn, name, detail, warn_only=warn)
    if rt.vllm_python.exists():
        good, detail = flashinfer_status(rt)
        row(good, "FlashInfer kernels", detail)
    missing = missing_setup_tools()
    row(
        not missing,
        "uv",
        "found" if not missing else f"missing: {', '.join(missing)}",
        warn_only=True,
    )
    ports = [8100, 8101, 8102]
    busy = [p for p in ports if not hardware.port_is_free(p)]
    row(not busy, "Ports", "8100-8102 free" if not busy else f"in use: {busy}", warn_only=True)
    if model:
        try:
            info = inspect_model(model)
            row(True, "Model", f"{info.model_type}, {info.num_layers} layers, {info.size_label}")
            from .compat import known_issues

            for note in known_issues(info):
                row(False, "Model note", note, warn_only=True)
        except ModelInfoError as e:
            row(False, "Model", str(e))
    console.print(t)
    if not ok:
        raise typer.Exit(1)


@app.command()
def plan(
    model: ModelOpt,
    preset: PresetOpt = "standard",
    data: DataOpt = None,
    public_data: PublicOpt = DEFAULT_PUBLIC_DATASET,
    workdir: WorkdirOpt = None,
    gpus: GpusOpt = None,
    samples: Annotated[int | None, typer.Option(help="Override the number of samples.")] = None,
    epochs: Annotated[int | None, typer.Option(help="Override epochs.")] = None,
    seq_len: Annotated[int | None, typer.Option(help="Override sequence length.")] = None,
    draft_layers: Annotated[int | None, typer.Option(help="Override drafter depth.")] = None,
    eval_prompts: Annotated[str | None, typer.Option(help="JSONL prompts for eval.")] = None,
    trust_remote_code: Annotated[
        bool | None, typer.Option(help="Allow models with custom code.")
    ] = None,
) -> None:
    """Work out a training plan for a model and save it to plan.yaml."""
    workdir = workdir or _default_workdir(model, preset)
    p = _build_plan(
        model,
        preset,
        data,
        public_data,
        workdir,
        gpus,
        samples,
        epochs,
        seq_len,
        draft_layers,
        eval_prompts,
        trust_remote_code,
    )
    print_plan(p)
    path = Paths(workdir).plan
    p.save(path)
    console.print(f"\nSaved to {path}. Edit it if you like, then: {_resume_hint(workdir)}")


@app.command(name="run")
def run_cmd(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to train for.")] = None,
    plan_file: Annotated[
        Path | None, typer.Option("--plan", help="Use an existing plan.yaml.")
    ] = None,
    workdir: WorkdirOpt = None,
    preset: Annotated[
        str | None,
        typer.Option(help=f"How much training to do: {', '.join(PRESETS)} (default: standard)."),
    ] = None,
    data: DataOpt = None,
    public_data: PublicOpt = DEFAULT_PUBLIC_DATASET,
    gpus: GpusOpt = None,
    samples: Annotated[int | None, typer.Option(help="Override the number of samples.")] = None,
    epochs: Annotated[int | None, typer.Option(help="Override epochs.")] = None,
    seq_len: Annotated[int | None, typer.Option(help="Override sequence length.")] = None,
    draft_layers: Annotated[int | None, typer.Option(help="Override drafter depth.")] = None,
    eval_prompts: Annotated[str | None, typer.Option(help="JSONL prompts for eval.")] = None,
    trust_remote_code: Annotated[
        bool | None, typer.Option(help="Allow models with custom code.")
    ] = None,
    from_stage: Annotated[
        str | None, typer.Option(help=f"Redo from this stage: {', '.join(STAGES)}.")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print every command instead of running.")] = False,
    replan: Annotated[
        bool, typer.Option(help="Rebuild plan.yaml even if the work folder already has one.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before starting.")] = False,
    quiet: Annotated[bool, typer.Option(help="Only write subprocess output to logs.")] = False,
    home: HomeOpt = None,
) -> None:
    """Plan (if needed) and run the whole pipeline. Re-run to resume."""
    if gpus and not dry_run:
        console.print("[red]--gpus is only for planning and --dry-run.[/]")
        raise typer.Exit(2)
    if not dry_run:
        _require_linux("Training")

    if model and not workdir and not plan_file:
        workdir = _default_workdir(model, preset or "standard")
    if workdir is None and plan_file is not None:
        workdir = plan_file.parent
    if workdir is None:
        console.print("[red]Pass --model, --plan, or --workdir of an earlier run.[/]")
        raise typer.Exit(2)

    # One run per work folder; status, serve and --dry-run don't take the lock.
    try:
        lock = contextlib.nullcontext() if dry_run else WorkdirLock(Paths(workdir).lock)
        with lock:
            _run(
                model=model,
                plan_file=plan_file,
                workdir=workdir,
                preset=preset,
                data=data,
                public_data=public_data,
                gpus=gpus,
                samples=samples,
                epochs=epochs,
                seq_len=seq_len,
                draft_layers=draft_layers,
                eval_prompts=eval_prompts,
                trust_remote_code=trust_remote_code,
                from_stage=from_stage,
                dry_run=dry_run,
                replan=replan,
                yes=yes,
                quiet=quiet,
                home=home,
            )
    except WorkdirBusy as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e


def _check_saved_plan(p: Plan, plan_file: Path, model: str | None, preset: str | None) -> None:
    """Refuse to silently train for a different model or preset than the saved plan."""
    problems = []
    if model and _model_ref(model) != p.model.path:
        problems.append(f"--model {model} (the plan is for {p.model.path})")
    if preset and preset != p.preset:
        problems.append(f"--preset {preset} (the plan uses {p.preset})")
    if problems:
        console.print(
            f"[red]{plan_file} was made for something else:[/] {'; '.join(problems)}.\n"
            "Pass --replan to plan again in this folder (earlier results are moved aside), "
            "or use a different --workdir."
        )
        raise typer.Exit(2)


def _run(
    *,
    model: str | None,
    plan_file: Path | None,
    workdir: Path,
    preset: str | None,
    data: list[str] | None,
    public_data: str,
    gpus: str | None,
    samples: int | None,
    epochs: int | None,
    seq_len: int | None,
    draft_layers: int | None,
    eval_prompts: str | None,
    trust_remote_code: bool | None,
    from_stage: str | None,
    dry_run: bool,
    replan: bool,
    yes: bool,
    quiet: bool,
    home: Path | None,
) -> None:
    existing = Paths(workdir).plan
    if plan_file is None and existing.exists() and not replan:
        # Resuming: keep the saved plan (and any edits to it) instead of re-planning.
        plan_file = existing
        given = {
            "--data": data,
            "--public-data": None if public_data == DEFAULT_PUBLIC_DATASET else public_data,
            "--samples": samples,
            "--epochs": epochs,
            "--seq-len": seq_len,
            "--draft-layers": draft_layers,
            "--eval-prompts": eval_prompts,
            "--gpus": gpus,
            "--trust-remote-code": trust_remote_code,
        }
        ignored = [flag for flag, v in given.items() if v is not None]
        if ignored:
            console.print(
                f"[yellow]note:[/] using the existing {existing}; {', '.join(ignored)} "
                "ignored. Pass --replan to rebuild it."
            )
    if plan_file is not None:
        p = Plan.load(plan_file)
        _check_saved_plan(p, plan_file, model, preset)
        if plan_file.resolve() != existing.resolve():
            p.save(existing)
    elif model:
        p = _build_plan(
            model,
            preset or "standard",
            data,
            public_data,
            workdir,
            gpus,
            samples,
            epochs,
            seq_len,
            draft_layers,
            eval_prompts,
            trust_remote_code,
        )
        p.save(existing)
    else:
        console.print(f"[red]No plan.yaml in {workdir}.[/] Pass --model to plan a new run.")
        raise typer.Exit(2)

    print_plan(p)
    valid = stages_for(p)
    if from_stage is not None and from_stage not in valid:
        extra = " (it only exists in offline mode)" if from_stage == "hidden" else ""
        console.print(
            f"[red]Unknown --from-stage {from_stage!r}{extra}.[/] Stages in this plan: "
            f"{', '.join(valid)}."
        )
        raise typer.Exit(2)
    problems = p.problems()
    for prob in problems:
        console.print(f"[red]plan.yaml problem:[/] {prob}")
    if (p.errors or problems) and not dry_run:
        console.print("[red]Fix the blockers above (or edit plan.yaml) before running.[/]")
        raise typer.Exit(2)

    rt = _runtime(home)
    if not dry_run:
        bad = [(n, d) for n, good, d in rt.check() if not good]
        if bad:
            for n, d in bad:
                console.print(f"[red]{n}:[/] {d}")
            raise typer.Exit(2)
        if not yes and not typer.confirm("Start?", default=True):
            raise typer.Exit()

    pipe = Pipeline(p, rt, workdir, console, dry_run=dry_run, echo=not quiet)
    if not dry_run:
        install_signal_handlers()  # SIGTERM/SIGHUP stop child processes like Ctrl+C
    try:
        pipe.run(from_stage=from_stage)
    except StepFailed as e:
        _fail(e, workdir)
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        _say(f"\nStopped. Resume with: {_resume_hint(workdir)}")
        raise typer.Exit(130) from None
    if not dry_run:
        console.print(f"\n[green bold]All done.[/] See {Paths(workdir).report_dir / 'report.md'}")


@app.command()
def status(workdir: Annotated[Path, typer.Option(help="Run folder.")]) -> None:
    """Show which stages of a run are done."""
    paths = Paths(workdir)
    if not paths.plan.exists():
        console.print(f"[red]No plan.yaml in {workdir}.[/]")
        raise typer.Exit(2)
    p = Plan.load(paths.plan)
    state = State(paths.state)
    pipe = Pipeline(p, _runtime(None), workdir, console, dry_run=True)
    t = Table(title=f"Run {workdir}", title_justify="left")
    t.add_column("Stage")
    t.add_column("Status")
    t.add_column("Finished")
    for s in pipe.stages():
        rec = state.get(s) or {}
        st = rec.get("status", "pending")
        if st == "done" and not state.done(s, pipe.hashes[s]):
            st = "stale (plan changed)"
        color = {"done": "green", "failed": "red"}.get(st, "yellow")
        t.add_row(s, f"[{color}]{st}[/]", rec.get("finished_at", ""))
    console.print(t)


@app.command()
def serve(workdir: Annotated[Path, typer.Option(help="Run folder.")]) -> None:
    """Print the vLLM command that serves your model with the trained drafter."""
    paths = Paths(workdir)
    if not paths.plan.exists():
        console.print(f"[red]No plan.yaml in {workdir}.[/]")
        raise typer.Exit(2)
    p = Plan.load(paths.plan)
    rec = (State(paths.state).get("train") or {}).get("info") or {}
    drafter = find_drafter(paths.checkpoints)
    if drafter is None and rec.get("drafter"):
        recorded = Path(rec["drafter"])
        if not (recorded / "config.json").exists():
            console.print(
                f"[red]The drafter recorded in state.json ({recorded}) is gone.[/] Train again "
                f"with: {_resume_hint(workdir)} --from-stage train"
            )
            raise typer.Exit(2)
        drafter = recorded
    if drafter is None:
        console.print(
            f"[red]No trained drafter found yet in {paths.checkpoints}.[/] Run training first: "
            f"{_resume_hint(workdir)}"
        )
        raise typer.Exit(2)
    for title, cmd in commands.serve_commands(p, drafter):
        console.print(f"[bold]{title}:[/]")
        console.print(cmd, soft_wrap=True, markup=False, highlight=False)
        console.print()


if __name__ == "__main__":
    app()
