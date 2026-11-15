"""Run the training pipeline stage by stage, resumably.

Stages: regen -> prepare -> [hidden] -> train -> eval -> report.
Each finished stage is recorded in state.json with a hash of the plan
fields it depends on. Re-running skips stages that are still valid and
redoes a stage (and everything after it) when its inputs changed.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import httpx
from rich.console import Console

from . import bench, commands, report
from .commands import Cmd
from .planner import Plan
from .procs import (
    TRAIN_STOP_GRACE,
    StepFailed,
    VllmServer,
    log_size,
    read_from,
    run,
    tail_file,
)
from .runtime import Runtime

STAGES = ("regen", "prepare", "hidden", "train", "eval", "report")
_OOM = re.compile(r"OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED")
MIN_RETRY_ANCHORS = 256


class Paths:
    def __init__(self, workdir: Path):
        workdir = workdir.expanduser().resolve()
        self.root = workdir
        self.plan = workdir / "plan.yaml"
        self.state = workdir / "state.json"
        self.logs = workdir / "logs"
        self.regen = workdir / "data" / "regen"
        self.prepared = workdir / "data" / "prepared"
        self.hs_stream = workdir / "hs_stream"
        self.hs_tmp = workdir / "hs_tmp"
        self.hs_cache = workdir / "hidden_states"
        self.checkpoints = workdir / "checkpoints"
        self.report_dir = workdir / "report"


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def stage_hashes(plan: Plan) -> dict[str, str]:
    """Each stage hashes its own inputs plus everything upstream of it."""
    regen = {
        "model": plan.model.path,
        "sources": [s.model_dump() for s in plan.data.sources],
        "max_tokens": plan.data.regen_max_tokens,
        "context": plan.data.regen_max_model_len,
    }
    prepare = {
        **regen,
        "seq_len": plan.data.seq_len,
        "draft_vocab": plan.draft.draft_vocab_size,
    }
    hidden = {**prepare, "layers": plan.draft.target_layer_ids, "mode": plan.layout.mode}
    train = {
        **hidden,
        "draft": plan.draft.model_dump(),
        "train": plan.train.model_dump(),
        "layout": plan.layout.model_dump(),
    }
    ev = {**train, "eval": plan.eval.model_dump()}
    return {
        "regen": _digest(regen),
        "prepare": _digest(prepare),
        "hidden": _digest(hidden),
        "train": _digest(train),
        "eval": _digest(ev),
        "report": _digest(ev),
    }


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"stages": {}}
        if path.exists():
            self.data = json.loads(path.read_text())

    def get(self, stage: str) -> dict | None:
        return self.data["stages"].get(stage)

    def done(self, stage: str, digest: str) -> bool:
        rec = self.get(stage)
        return bool(rec and rec.get("status") == "done" and rec.get("hash") == digest)

    def mark(self, stage: str, digest: str, info: dict | None = None) -> None:
        self.data["stages"][stage] = {
            "status": "done",
            "hash": digest,
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "info": info or {},
        }
        self._save()

    def fail(self, stage: str, message: str) -> None:
        rec = self.data["stages"].setdefault(stage, {})
        rec.update({"status": "failed", "error": message})
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def find_drafter(checkpoints: Path) -> Path | None:
    """checkpoint_best if present, else the highest epoch folder Speculators wrote."""
    best = checkpoints / "checkpoint_best"
    if (best / "config.json").exists():
        # Keep the symlink name: it follows the best epoch if training resumes.
        return best.absolute()
    if not checkpoints.is_dir():
        return None
    # Speculators names epoch folders by integer (<save>/<N>/); other names are
    # readability symlinks or interrupted checkpoints.
    epochs = [p for p in checkpoints.iterdir() if p.name.isdigit() and (p / "config.json").exists()]
    if epochs:
        return max(epochs, key=lambda p: int(p.name)).resolve()
    others = [p.parent for p in checkpoints.glob("*/config.json")]
    return max(others, key=lambda p: p.stat().st_mtime).resolve() if others else None


def count_regenerated(path: Path) -> int:
    """Distinct conversations already in a regen output file (rows are per turn)."""
    if not path.exists():
        return 0
    seen: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("primary_id") or row.get("id")
            if key is not None:
                seen.add(str(key))
    return len(seen)


class Pipeline:
    def __init__(
        self,
        plan: Plan,
        rt: Runtime,
        workdir: Path,
        console: Console,
        *,
        dry_run: bool = False,
        echo: bool = True,
        max_oom_retries: int = 3,
    ):
        self.plan, self.rt, self.console = plan, rt, console
        self.paths = Paths(workdir)
        self.state = State(self.paths.state)
        self.dry_run, self.echo = dry_run, echo
        self.max_oom_retries = max_oom_retries
        self.hashes = stage_hashes(plan)
        self._fresh = False  # set while a stage is forced with --from-stage

    # ----------------------------------------------------------------- helpers

    def stages(self) -> list[str]:
        return [s for s in STAGES if s != "hidden" or self.plan.layout.mode == "offline"]

    def _exec(self, cmd: Cmd, log_name: str, grace: float | None = None) -> None:
        if self.dry_run:
            self.console.print(f"[bold]# {cmd.label}[/]\n{cmd.shell()}\n", soft_wrap=True)
            return
        kwargs = {"grace": grace} if grace else {}
        run(cmd, self.paths.logs / log_name, self.console, echo=self.echo, **kwargs)

    def _server(self, cmd: Cmd, port: int, log_name: str):
        if self.dry_run:
            self.console.print(
                f"[bold]# start in background: {cmd.label}[/]\n{cmd.shell()}\n", soft_wrap=True
            )
            return contextlib.nullcontext()
        return VllmServer(cmd, port, self.paths.logs / log_name, self.console)

    def _claim_dir(self, path: Path, digest: str) -> None:
        """Make sure `path` only holds outputs made from these inputs.

        Upstream steps resume from whatever is already in their output folder
        (regen appends, offline generation skips existing files, training loads
        the last checkpoint). That is what we want after a crash, but not after
        the plan changed or when a stage is forced, so then the old folder is
        moved aside. Nothing is deleted.
        """
        if self.dry_run:
            return
        marker = self.paths.root / ".inputs" / path.name
        old = marker.read_text().strip() if marker.exists() else None
        stale = self._fresh or (old is not None and old != digest)
        if stale and path.exists() and any(path.iterdir()):
            aside = path.with_name(f"{path.name}.stale-{dt.datetime.now():%Y%m%d-%H%M%S}")
            path.rename(aside)
            self.console.print(
                f"[yellow]![/] moved the previous {path.name} folder to {aside} "
                "(delete it when you no longer need it)"
            )
        path.mkdir(parents=True, exist_ok=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(digest)

    # ----------------------------------------------------------------- driver

    def run(self, from_stage: str | None = None, only: list[str] | None = None) -> None:
        order = self.stages()
        if from_stage and from_stage not in order:
            raise ValueError(f"Unknown stage {from_stage!r}; stages are {order}")
        start = order.index(from_stage) if from_stage else 0
        dirty = False
        for i, stage in enumerate(order):
            if (only and stage not in only) or i < start:
                continue
            forced = from_stage is not None
            if not forced and not dirty and self.state.done(stage, self.hashes[stage]):
                self.console.print(f"[green]✓[/] {stage}: already done, skipping")
                continue
            dirty = True
            self._fresh = forced
            self.console.rule(f"[bold]{stage}")
            try:
                info = getattr(self, f"_stage_{stage}")()
            except StepFailed as e:
                if not self.dry_run:
                    self.state.fail(stage, str(e))
                raise
            finally:
                self._fresh = False
            if not self.dry_run:
                self.state.mark(stage, self.hashes[stage], info)

    # ----------------------------------------------------------------- stages

    def _regen_file(self, slug: str) -> Path:
        return self.paths.regen / f"{slug}.jsonl"

    def _stage_regen(self) -> dict:
        plan, rt = self.plan, self.rt
        self._claim_dir(self.paths.regen, self.hashes["regen"])
        rows: dict[str, int] = {}
        todo = []
        for src in plan.data.sources:
            out = self._regen_file(src.slug)
            done = 0 if self.dry_run else count_regenerated(out)
            if done >= src.limit:
                self.console.print(f"[green]✓[/] {src.slug}: {done} conversations already done")
                rows[src.slug] = done
            else:
                # --resume skips finished conversations without counting them
                # toward --limit, so ask only for what is still missing.
                todo.append((src, out, src.limit - done))
        if todo:
            with self._server(
                commands.regen_server(plan, rt), plan.ports.regen, "regen_server.log"
            ):
                for src, out, remaining in todo:
                    log = f"regen_{src.slug}.log"
                    self._exec(commands.regen(plan, rt, src, out, limit=remaining), log)
                    if self.dry_run:
                        continue
                    n = count_regenerated(out)
                    if n == 0:
                        raise StepFailed(
                            f"Response regeneration for {src.value} produced no rows",
                            self.paths.logs / log,
                            tail_file(self.paths.logs / log),
                        )
                    if n < 0.5 * src.limit:
                        self.console.print(
                            f"[yellow]![/] only {n} of {src.limit} conversations regenerated "
                            f"for {src.value}; see {self.paths.logs / log}"
                        )
                    rows[src.slug] = n
        return {"conversations": rows}

    def _stage_prepare(self) -> dict:
        self._claim_dir(self.paths.prepared, self.hashes["prepare"])
        files = [self._regen_file(s.slug) for s in self.plan.data.sources]
        self._exec(commands.prepare(self.plan, self.rt, files, self.paths.prepared), "prepare.log")
        return {"prepared": str(self.paths.prepared)}

    def _stage_hidden(self) -> dict:
        plan, rt, p = self.plan, self.rt, self.paths
        self._claim_dir(p.hs_cache, self.hashes["hidden"])
        self._claim_dir(p.hs_tmp, self.hashes["hidden"])
        with self._server(
            commands.extract_server(plan, rt, p.hs_tmp), plan.ports.extract, "extract_server.log"
        ):
            self._exec(commands.generate_offline(plan, rt, p.prepared, p.hs_cache), "hidden.log")
        return {"hidden_states": str(p.hs_cache)}

    def _stage_train(self) -> dict:
        plan, rt, p = self.plan, self.rt, self.paths
        self._claim_dir(p.checkpoints, self.hashes["train"])
        online = plan.layout.mode == "online"
        server = (
            self._server(
                commands.extract_server(plan, rt, p.hs_stream),
                plan.ports.extract,
                "extract_server.log",
            )
            if online
            else contextlib.nullcontext()
        )
        log = p.logs / "train.log"
        anchors = plan.draft.max_anchors
        # Server failures propagate as they are; only the train command's own
        # failures are checked for out-of-memory and retried.
        with server:
            for attempt in range(self.max_oom_retries + 1):
                cmd = commands.train(
                    plan,
                    rt,
                    p.prepared,
                    p.checkpoints,
                    hs_cache=None if online else p.hs_cache,
                    max_anchors=anchors,
                )
                offset = log_size(log)
                try:
                    self._exec(cmd, "train.log", grace=TRAIN_STOP_GRACE)
                    break
                except StepFailed as e:
                    out_of_memory = bool(_OOM.search(read_from(log, offset)))
                    can_shrink = anchors // 2 >= MIN_RETRY_ANCHORS
                    if out_of_memory and can_shrink and attempt < self.max_oom_retries:
                        anchors //= 2
                        self.console.print(
                            f"[yellow]![/] training ran out of GPU memory; retrying with "
                            f"max_anchors={anchors}"
                        )
                        continue
                    if out_of_memory:
                        raise StepFailed(
                            "Training ran out of GPU memory even after shrinking max_anchors. "
                            "Lower data.seq_len or draft.num_layers in plan.yaml.",
                            e.log,
                            e.tail,
                        ) from e
                    raise
        if self.dry_run:
            return {}
        drafter = find_drafter(p.checkpoints)
        if drafter is None:
            raise StepFailed(f"Training finished but no checkpoint was found in {p.checkpoints}")
        self.console.print(f"[green]✓[/] drafter saved at {drafter}")
        return {"drafter": str(drafter), "max_anchors": anchors}

    def _drafter(self) -> Path | None:
        found = find_drafter(self.paths.checkpoints)
        if found:
            return found
        path = ((self.state.get("train") or {}).get("info") or {}).get("drafter")
        return Path(path) if path else None

    def _stage_eval(self) -> dict:
        plan, rt = self.plan, self.rt
        drafter = self._drafter() or (self.paths.checkpoints / "checkpoint_best")
        prompts = bench.load_prompts(plan.eval.prompts_file, plan.eval.num_prompts)
        if not prompts:
            raise StepFailed("No usable eval prompts found.")
        results = {}
        for name, d in (("baseline", None), ("dspark", drafter)):
            server = commands.eval_server(plan, rt, d)
            log = f"eval_{name}_server.log"
            with self._server(server, plan.ports.eval, log):
                if self.dry_run:
                    self.console.print(
                        f"# benchmark {len(prompts)} prompts against {server.label}\n"
                    )
                    continue
                self.console.print(f"  benchmarking {len(prompts)} prompts ({name})")
                try:
                    results[name] = bench.run_bench(
                        commands.base_url(plan.ports.eval),
                        prompts,
                        max_tokens=plan.eval.max_tokens,
                        temperature=plan.eval.temperature,
                    )
                except (bench.BenchError, httpx.HTTPError) as e:
                    path = self.paths.logs / log
                    raise StepFailed(
                        f"Benchmark ({name}) failed: {e}", path, tail_file(path)
                    ) from e
                if results[name].get("failed"):
                    self.console.print(
                        f"[yellow]![/] {results[name]['failed']} prompts failed ({name}) and were "
                        "left out"
                    )
        if self.dry_run:
            return {}
        results["compare"] = bench.compare(results["baseline"], results["dspark"])
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)
        (self.paths.report_dir / "eval.json").write_text(json.dumps(results, indent=2))
        c = results["compare"]
        self.console.print(
            f"[green]✓[/] speedup {c['speedup']}x, mean acceptance length {c['acceptance_length']}"
        )
        return results

    def _stage_report(self) -> dict:
        if self.dry_run:
            self.console.print(f"# write {self.paths.report_dir / 'report.md'}")
            return {}
        eval_rec = (self.state.get("eval") or {}).get("info") or None
        drafter = self._drafter()
        text = report.render(self.plan, drafter, eval_rec)
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)
        out = self.paths.report_dir / "report.md"
        out.write_text(text, encoding="utf-8")
        self.console.print(f"[green]✓[/] report written to {out}")
        return {"report": str(out)}
