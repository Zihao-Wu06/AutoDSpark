import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from autodspark import SPECULATORS_VERSION, VLLM_VERSION, runtime
from autodspark.runtime import Runtime


def _rt(tmp_path: Path, **kw) -> Runtime:
    return Runtime.resolve(home=tmp_path / "home", **kw)


def _fake_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


# --------------------------------------------------------------------------- setup commands


def test_setup_pins_the_tested_vllm(tmp_path):
    cmds = runtime.setup_commands(_rt(tmp_path))
    vllm_install = cmds[-1]
    assert f"vllm=={VLLM_VERSION}" in vllm_install
    assert f"hs-connectors=={SPECULATORS_VERSION}" in vllm_install
    assert not any(">=" in arg for c in cmds for arg in c)


def test_setup_vllm_override_is_kept(tmp_path):
    cmds = runtime.setup_commands(_rt(tmp_path), vllm_spec="vllm==0.34.1")
    assert "vllm==0.34.1" in cmds[-1]
    assert f"vllm=={VLLM_VERSION}" not in cmds[-1]


def test_setup_creates_missing_venvs(tmp_path):
    rt = _rt(tmp_path)
    venvs = [c for c in runtime.setup_commands(rt, python_version="3.11") if c[1] == "venv"]
    assert venvs == [
        ["uv", "venv", str(rt.speculators_python.parent.parent), "--python", "3.11"],
        ["uv", "venv", str(rt.vllm_python.parent.parent), "--python", "3.11"],
    ]


def test_setup_rerun_reuses_existing_venvs(tmp_path):
    rt = _rt(tmp_path)
    for py in (rt.speculators_python, rt.vllm_python):
        env = py.parent.parent
        (env / "bin").mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("home = /usr/bin\n")
        (env / "bin" / "python").write_text("")
    cmds = runtime.setup_commands(rt)
    assert not [c for c in cmds if c[1] == "venv"]
    # the installs still run, so a retry after a network failure finishes the job
    assert [c[:3] for c in cmds] == [["uv", "pip", "install"]] * 2


def test_setup_rerun_with_only_one_env(tmp_path):
    rt = _rt(tmp_path)
    spec_env = rt.speculators_python.parent.parent
    (spec_env / "bin").mkdir(parents=True)
    (spec_env / "pyvenv.cfg").write_text("")
    (spec_env / "bin" / "python").write_text("")
    venvs = [c for c in runtime.setup_commands(rt) if c[1] == "venv"]
    assert [c[2] for c in venvs] == [str(rt.vllm_python.parent.parent)]


def test_setup_rebuilds_a_broken_venv(tmp_path):
    """A half-built env (pyvenv.cfg but no interpreter) is removed and recreated."""
    rt = _rt(tmp_path)
    env = rt.vllm_python.parent.parent
    env.mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("")
    cmds = runtime.setup_commands(rt)
    i = cmds.index(["rm", "-rf", str(env)])
    assert cmds[i + 1][:3] == ["uv", "venv", str(env)]


# --------------------------------------------------------------------------- speculators source


def _archive(with_script: bool = True) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        name = "speculators-0.9.0/scripts/launch_vllm.py" if with_script else "speculators/x.py"
        data = b"print('hi')"
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeStream:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        yield self.data


@pytest.fixture
def fake_download(monkeypatch):
    calls = []

    def use(data: bytes):
        def stream(method, url, **kw):
            calls.append(url)
            return _FakeStream(data)

        monkeypatch.setattr(runtime.httpx, "stream", stream)
        return calls

    return use


def test_fetch_replaces_managed_copy(tmp_path, fake_download):
    rt = _rt(tmp_path)
    assert rt.speculators_src_managed
    rt.speculators_src.mkdir(parents=True)
    (rt.speculators_src / "stale.txt").write_text("old")
    fake_download(_archive())
    runtime.fetch_speculators_src(rt)
    assert rt.launch_vllm.read_text() == "print('hi')"
    assert not (rt.speculators_src / "stale.txt").exists()
    assert [p.name for p in rt.speculators_src.parent.iterdir()] == ["speculators-src"]


def test_fetch_never_deletes_user_directory(tmp_path, fake_download, monkeypatch):
    mine = tmp_path / "my-speculators"
    mine.mkdir()
    (mine / "notes.py").write_text("precious")
    monkeypatch.setenv("AUTODSPARK_SPECULATORS_SRC", str(mine))
    rt = _rt(tmp_path)
    assert not rt.speculators_src_managed
    calls = fake_download(_archive())
    with pytest.raises(RuntimeError, match="AUTODSPARK_SPECULATORS_SRC"):
        runtime.fetch_speculators_src(rt)
    assert (mine / "notes.py").read_text() == "precious"
    assert calls == []
    assert [p.name for p in tmp_path.iterdir() if p.name != "home"] == ["my-speculators"]


def test_fetch_fills_empty_user_directory(tmp_path, fake_download):
    mine = tmp_path / "empty"
    mine.mkdir()
    rt = _rt(tmp_path, speculators_src=str(mine))
    fake_download(_archive())
    runtime.fetch_speculators_src(rt)
    assert (mine / "scripts" / "launch_vllm.py").exists()


def test_fetch_bad_archive_leaves_no_temp_dir(tmp_path, fake_download):
    rt = _rt(tmp_path)
    fake_download(_archive(with_script=False))
    with pytest.raises(RuntimeError, match="launch_vllm"):
        runtime.fetch_speculators_src(rt)
    assert list(rt.speculators_src.parent.iterdir()) == []


def test_check_explains_bad_user_source(tmp_path):
    rt = _rt(tmp_path, speculators_src=str(tmp_path / "elsewhere"))
    rows = {name: (ok, detail) for name, ok, detail in rt.check()}
    ok, detail = rows["Speculators scripts"]
    assert not ok and "AUTODSPARK_SPECULATORS_SRC" in detail


# --------------------------------------------------------------------------- environment check


def _check(rt, monkeypatch, versions, cuda=(True, "cuda ok")):
    monkeypatch.setattr(runtime, "_module_version", lambda py, mod: versions.get(mod))
    monkeypatch.setattr(runtime, "cuda_status", lambda py: cuda)
    return {name: (ok, detail) for name, ok, detail in rt.check()}


def test_check_accepts_tested_vllm(tmp_path, monkeypatch):
    rt = _rt(tmp_path)
    _fake_python(rt.speculators_python)
    _fake_python(rt.vllm_python)
    rows = _check(
        rt,
        monkeypatch,
        {"speculators": "0.9.0", "vllm": f"{VLLM_VERSION}+cu129", "hs_connectors": "0.9.0"},
    )
    assert rows["vLLM env"] == (True, f"{VLLM_VERSION}+cu129")
    assert rows["CUDA in vLLM env"] == (True, "cuda ok")


def test_check_warns_on_untested_vllm(tmp_path, monkeypatch):
    rt = _rt(tmp_path)
    _fake_python(rt.vllm_python)
    rows = _check(rt, monkeypatch, {"vllm": "0.40.0", "hs_connectors": "0.9.0"})
    ok, detail = rows["vLLM env"]
    assert ok  # a warning, so `run` is not blocked
    assert "0.40.0" in detail and "warning" in detail and VLLM_VERSION in detail


def test_check_skips_cuda_probe_without_env(tmp_path, monkeypatch):
    rt = _rt(tmp_path)

    def boom(py):
        raise AssertionError("must not probe a missing env")

    monkeypatch.setattr(runtime, "cuda_status", boom)
    rows = {name: ok for name, ok, _ in rt.check()}
    assert rows["vLLM env"] is False
    assert "CUDA in vLLM env" not in rows


def test_check_reports_cuda_failure(tmp_path, monkeypatch):
    rt = _rt(tmp_path)
    _fake_python(rt.vllm_python)
    rows = _check(rt, monkeypatch, {"vllm": VLLM_VERSION}, cuda=(False, "driver too old"))
    assert rows["CUDA in vLLM env"] == (False, "driver too old")


def _probe(monkeypatch, stdout="", stderr="", returncode=0, exc=None):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw.get("timeout")
        if exc:
            raise exc
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    return seen


def test_cuda_status_reports_capability(monkeypatch):
    info = {
        "torch": "2.9.0+cu129",
        "cuda": "12.9",
        "available": True,
        "devices": [["NVIDIA H100", 9, 0], ["NVIDIA H100", 9, 0]],
    }
    seen = _probe(monkeypatch, stdout=json.dumps(info) + "\n")
    ok, detail = runtime.cuda_status(Path("/env/bin/python"))
    assert ok
    assert detail == "torch 2.9.0+cu129 (CUDA 12.9): 2x NVIDIA H100 (compute 9.0)"
    assert seen["cmd"][0] == "/env/bin/python" and seen["timeout"]


def test_cuda_status_driver_too_old(monkeypatch):
    info = {"torch": "2.9.0+cu129", "cuda": "12.9", "available": False}
    stderr = (
        "/env/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: The NVIDIA driver "
        "on your system is too old (found version 11040).\n"
        "  return torch._C._cuda_getDeviceCount() > 0\n"
    )
    _probe(monkeypatch, stdout=json.dumps(info), stderr=stderr)
    ok, detail = runtime.cuda_status(Path("python"))
    assert not ok
    assert "cannot use the GPU" in detail and "driver" in detail
    assert "found version 11040" in detail


def test_cuda_status_cpu_only_torch(monkeypatch):
    info = {"torch": "2.9.0+cpu", "cuda": None, "available": False}
    _probe(monkeypatch, stdout=json.dumps(info))
    ok, detail = runtime.cuda_status(Path("python"))
    assert not ok and "CPU-only" in detail


def test_cuda_status_torch_import_fails(monkeypatch):
    _probe(monkeypatch, stderr="ModuleNotFoundError: No module named 'torch'\n", returncode=1)
    ok, detail = runtime.cuda_status(Path("python"))
    assert not ok and "No module named 'torch'" in detail


def test_cuda_status_timeout_is_not_fatal(monkeypatch):
    _probe(monkeypatch, exc=subprocess.TimeoutExpired("python", 60))
    ok, detail = runtime.cuda_status(Path("python"))
    assert ok and "not checked" in detail
