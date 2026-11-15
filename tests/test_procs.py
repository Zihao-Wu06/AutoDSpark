import os
import signal
import subprocess
import sys
import threading
import time

import pytest
from rich.console import Console

from autodspark import procs
from autodspark.commands import Cmd
from autodspark.procs import _group_alive, terminate_group

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")


@pytest.fixture
def restore_signals():
    saved = {s: signal.getsignal(s) for s in procs.STOP_SIGNALS}
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    for s, handler in saved.items():
        signal.signal(s, handler)


class _GoneTerminal:
    """stdout/console after an ssh drop: every write raises EIO."""

    def write(self, *a, **k):
        raise OSError(5, "Input/output error")

    def flush(self):
        raise OSError(5, "Input/output error")


def _broken_console():
    console = Console()
    console.print = lambda *a, **k: _GoneTerminal().write()
    return console


def _record_popen(monkeypatch):
    started = []
    real = subprocess.Popen

    def popen(*a, **k):
        started.append(real(*a, **k))
        return started[-1]

    monkeypatch.setattr(procs.subprocess, "Popen", popen)
    return started


@posix_only
def test_terminate_group_kills_children_even_if_parent_exited():
    # The parent exits at once and leaves a child behind in its process group,
    # like a crashed vLLM leaving engine workers on the GPU.
    proc = subprocess.Popen(
        ["sh", "-c", "sleep 60 & exit 0"],
        start_new_session=True,
    )
    proc.wait()
    time.sleep(0.2)
    assert _group_alive(proc.pid)
    terminate_group(proc, grace=5)
    assert not _group_alive(proc.pid)


@posix_only
def test_terminate_group_escalates_to_sigkill():
    proc = subprocess.Popen(
        ["sh", "-c", "trap '' TERM; sleep 60"],
        start_new_session=True,
    )
    time.sleep(0.2)
    start = time.time()
    terminate_group(proc, grace=1)
    assert not _group_alive(proc.pid)
    assert time.time() - start < 20


def _tar_with(entries):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = data
                tar.addfile(info)
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:gz")


def test_safe_extract_strips_top_dir_and_skips_links(tmp_path):
    from autodspark.runtime import _safe_extract

    tar = _tar_with(
        [
            ("speculators-0.9.0/scripts/launch_vllm.py", b"print('hi')", "file"),
            ("speculators-0.9.0/evil_link", "/etc/passwd", "symlink"),
        ]
    )
    _safe_extract(tar, tmp_path, strip=1)
    assert (tmp_path / "scripts" / "launch_vllm.py").read_text() == "print('hi')"
    assert not (tmp_path / "evil_link").exists()


def test_safe_extract_refuses_path_traversal(tmp_path):
    from autodspark.runtime import _safe_extract

    tar = _tar_with([("top/../../escape.txt", b"x", "file")])
    with pytest.raises(RuntimeError, match="Unsafe"):
        _safe_extract(tar, tmp_path / "dest", strip=1)


def test_child_env_activates_the_venv(tmp_path):
    from autodspark.commands import Cmd
    from autodspark.procs import _env

    venv = tmp_path / "envs" / "vllm"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    cmd = Cmd([str(venv / "bin" / "python"), "-m", "vllm"], gpus=[0])
    env = _env(cmd)
    assert env["PATH"].split(":")[0] == str(venv / "bin")  # ninja etc. are found
    assert env["VIRTUAL_ENV"] == str(venv)
    assert f"PATH={venv / 'bin'}:$PATH" in cmd.shell()

    plain = Cmd(["/usr/bin/python3", "-c", "pass"])
    assert plain.venv_bin is None
    assert "VIRTUAL_ENV" not in _env(plain) or _env(plain)["VIRTUAL_ENV"] != str(venv)


def test_flashinfer_status_without_kernels_or_nvcc(tmp_path, monkeypatch):
    from autodspark import runtime

    rt = runtime.Runtime.resolve(home=tmp_path)
    monkeypatch.setattr(runtime, "_module_version", lambda py, mod: None)
    monkeypatch.setattr(runtime, "find_nvcc", lambda: None)
    ok, detail = runtime.flashinfer_status(rt)
    assert not ok and "download-kernels" in detail
    monkeypatch.setattr(runtime, "find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    ok, detail = runtime.flashinfer_status(rt)
    assert ok and "compiles" in detail
    monkeypatch.setattr(runtime, "_module_version", lambda py, mod: "0.7.2")
    assert runtime.flashinfer_status(rt) == (True, "precompiled (flashinfer-jit-cache 0.7.2)")


@posix_only
@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_run_kills_child_on_signal_even_when_terminal_is_gone(
    sig, tmp_path, monkeypatch, restore_signals
):
    # ssh drop: SIGHUP arrives and every print raises EIO. The child (and its own
    # children) must still be stopped, or they keep holding GPUs and ports.
    procs.install_signal_handlers()
    started = _record_popen(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _GoneTerminal())
    cmd = Cmd(["sh", "-c", "echo up; sleep 60 & wait"], label="fake training")
    threading.Timer(0.5, os.kill, (os.getpid(), getattr(signal, sig))).start()
    with pytest.raises(KeyboardInterrupt):
        procs.run(cmd, tmp_path / "x.log", _broken_console(), grace=5)
    assert started and not _group_alive(started[0].pid)
    assert "up" in (tmp_path / "x.log").read_text()  # output still logged without a terminal


@posix_only
def test_second_interrupt_during_grace_still_sigkills(restore_signals):
    # The child ignores SIGTERM, so terminate_group waits out the grace period;
    # a second Ctrl+C during that wait must SIGKILL the group, not abandon it.
    proc = subprocess.Popen(["sh", "-c", "trap '' TERM; sleep 60"], start_new_session=True)
    time.sleep(0.2)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    start = time.time()
    with pytest.raises(KeyboardInterrupt):
        terminate_group(proc, grace=120)
    assert not _group_alive(proc.pid)
    assert time.time() - start < 20


@posix_only
def test_interrupts_are_deferred_during_final_kill(restore_signals):
    with pytest.raises(KeyboardInterrupt), procs.interrupts_deferred():
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.1)
        finished = True  # reached: the interrupt waited for the block
    assert finished
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@posix_only
def test_vllm_server_exit_kills_even_if_print_fails(tmp_path):
    server = procs.VllmServer(Cmd(["true"], label="vLLM"), 1, tmp_path / "s.log", _broken_console())
    server.proc = subprocess.Popen(["sh", "-c", "sleep 60 & wait"], start_new_session=True)
    pid = server.proc.pid
    time.sleep(0.2)
    server.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert not _group_alive(pid)
    assert server.proc is None


def test_procs_has_fallbacks_without_process_groups(monkeypatch):
    # Windows has no os.killpg / SIGKILL; stopping must fall back to the process itself.
    monkeypatch.setattr(procs, "_HAS_PROCESS_GROUPS", False)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    terminate_group(proc, grace=5)
    assert proc.poll() is not None


def test_signal_handlers_keep_nohup_ignore(restore_signals):
    """`nohup autodspark run` must survive an ssh hangup: an ignored SIGHUP stays ignored."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    procs.install_signal_handlers()
    assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
