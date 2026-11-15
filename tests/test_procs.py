import subprocess
import sys
import time

import pytest

from autodspark.procs import _group_alive, terminate_group


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
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


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
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
