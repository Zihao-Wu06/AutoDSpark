"""Run subprocesses with logs, and manage vLLM servers as context managers."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import httpx
from rich.console import Console

from .commands import Cmd, base_url
from .hardware import port_is_free

SERVER_STOP_GRACE = 60.0
# Speculators saves an "interrupted" checkpoint on SIGTERM; give it time.
TRAIN_STOP_GRACE = 300.0


class StepFailed(RuntimeError):
    def __init__(self, message: str, log: Path | None = None, tail: str = ""):
        super().__init__(message)
        self.log = log
        self.tail = tail


def tail_file(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    with path.open(encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=lines))


def read_from(path: Path, offset: int) -> str:
    """Text appended to `path` after byte `offset`."""
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", errors="replace")


def log_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _env(cmd: Cmd) -> dict[str, str]:
    env = dict(os.environ)
    env.update(cmd.env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if cmd.venv_bin:
        # Behave as if the env were activated, so tools it ships are found.
        env["PATH"] = f"{cmd.venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(cmd.venv_bin.parent)
    # Local servers must never go through an HTTP proxy.
    local = "127.0.0.1,localhost"
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = f"{env[key]},{local}" if env.get(key) else local
    return env


def _raise_interrupt(signum, frame):  # noqa: ARG001
    raise KeyboardInterrupt


def install_signal_handlers() -> None:
    """Treat SIGTERM/SIGHUP (e.g. a dropped ssh session) like Ctrl+C, so servers get stopped."""
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, _raise_interrupt)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_group(proc: subprocess.Popen, grace: float = SERVER_STOP_GRACE) -> None:
    """Stop a child and every process in its group (vLLM engine and TP workers).

    Children are started with start_new_session=True, so the group id is the
    child's pid. We signal the group even if the parent already exited, and
    wait until the whole group is gone so GPUs are free for the next step.
    """
    pgid = proc.pid
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline:
        proc.poll()  # reap the parent so it doesn't linger as a zombie
        if not _group_alive(pgid):
            return
        time.sleep(0.5)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)
    deadline = time.time() + 15
    while _group_alive(pgid) and time.time() < deadline:
        time.sleep(0.5)


def run(
    cmd: Cmd, log: Path, console: Console, *, echo: bool = True, grace: float = SERVER_STOP_GRACE
) -> None:
    """Run a command to completion, teeing output to `log`. Raises StepFailed."""
    log.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]>[/] {cmd.label}  [dim](log: {log})[/]")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {cmd.shell()}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd.argv,
            env=_env(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                if echo:
                    sys.stdout.write(line)
            code = proc.wait()
        except KeyboardInterrupt:
            console.print(f"  stopping {cmd.label} (up to {grace:.0f}s)...")
            terminate_group(proc, grace)
            raise
        # Leftover children (e.g. dataloader workers) must not hold GPUs.
        terminate_group(proc, grace=10)
    if code != 0:
        raise StepFailed(f"{cmd.label} failed (exit code {code})", log, tail_file(log))


class VllmServer:
    """Start a vLLM server in the background and stop it on exit.

    with VllmServer(cmd, port, log, console):
        ... talk to http://127.0.0.1:port ...
    """

    def __init__(
        self, cmd: Cmd, port: int, log: Path, console: Console, startup_timeout: float = 1800
    ):
        self.cmd, self.port, self.log, self.console = cmd, port, log, console
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return base_url(self.port)

    def __enter__(self) -> VllmServer:
        if not port_is_free(self.port):
            raise StepFailed(
                f"Port {self.port} is already in use. Stop whatever is using it (maybe a server "
                "left over from an earlier run), or change `ports` in plan.yaml."
            )
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.console.print(f"[bold cyan]>[/] starting {self.cmd.label}  [dim](log: {self.log})[/]")
        with self.log.open("a", encoding="utf-8") as f:
            f.write(f"\n$ {self.cmd.shell()}\n")
            f.flush()
            self.proc = subprocess.Popen(
                self.cmd.argv,
                env=_env(self.cmd),
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            self._wait_ready()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_ready(self) -> None:
        assert self.proc is not None
        start = time.time()
        while time.time() - start < self.startup_timeout:
            if self.proc.poll() is not None:
                raise StepFailed(
                    f"{self.cmd.label} exited during startup (code {self.proc.returncode})",
                    self.log,
                    tail_file(self.log),
                )
            try:
                r = httpx.get(f"{self.url}/health", timeout=5, trust_env=False)
                if r.status_code == 200:
                    self.console.print(f"  ready on {self.url} after {time.time() - start:.0f}s")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise StepFailed(
            f"{self.cmd.label} did not become ready within {self.startup_timeout:.0f}s",
            self.log,
            tail_file(self.log),
        )

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.console.print(f"  stopping {self.cmd.label}")
            terminate_group(self.proc)
            self.proc = None
