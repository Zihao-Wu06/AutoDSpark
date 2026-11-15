"""Where the heavy environments live, and how to create them.

Speculators and vLLM are installed into two separate virtualenvs, as the
Speculators docs recommend, so their torch pins never conflict. AutoDSpark
itself stays a light CLI that drives both.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx
from pydantic import BaseModel

from . import SPECULATORS_GIT_TAG, SPECULATORS_VERSION, VLLM_MIN_VERSION

SPECULATORS_ARCHIVE = "https://codeload.github.com/vllm-project/speculators/tar.gz/refs/tags"


def default_home() -> Path:
    return Path(os.environ.get("AUTODSPARK_HOME", "~/.cache/autodspark")).expanduser()


class Runtime(BaseModel):
    home: Path
    speculators_python: Path
    vllm_python: Path
    speculators_src: Path

    @classmethod
    def resolve(
        cls,
        home: Path | None = None,
        speculators_python: str | None = None,
        vllm_python: str | None = None,
        speculators_src: str | None = None,
    ) -> Runtime:
        home = (home or default_home()).expanduser()

        def pick(arg: str | None, env: str, default: Path) -> Path:
            value = arg or os.environ.get(env)
            return Path(value).expanduser() if value else default

        return cls(
            home=home,
            speculators_python=pick(
                speculators_python,
                "AUTODSPARK_SPECULATORS_PYTHON",
                home / "envs" / "speculators" / "bin" / "python",
            ),
            vllm_python=pick(
                vllm_python, "AUTODSPARK_VLLM_PYTHON", home / "envs" / "vllm" / "bin" / "python"
            ),
            speculators_src=pick(
                speculators_src, "AUTODSPARK_SPECULATORS_SRC", home / "speculators-src"
            ),
        )

    @property
    def launch_vllm(self) -> Path:
        return self.speculators_src / "scripts" / "launch_vllm.py"

    def check(self) -> list[tuple[str, bool, str]]:
        """(name, ok, detail) rows for `autodspark doctor`."""
        rows = []
        for label, py, mod in (
            ("Speculators env", self.speculators_python, "speculators"),
            ("vLLM env", self.vllm_python, "vllm"),
        ):
            if not py.exists():
                rows.append((label, False, f"{py} not found. Run `autodspark setup`."))
                continue
            version = _module_version(py, mod)
            rows.append((label, version is not None, version or f"`import {mod}` failed in {py}"))
        if self.vllm_python.exists():
            hs = _module_version(self.vllm_python, "hs_connectors")
            rows.append(
                (
                    "hs_connectors in vLLM env",
                    hs is not None,
                    hs or "missing; install hs-connectors into the vLLM env",
                )
            )
        rows.append(
            (
                "Speculators scripts",
                self.launch_vllm.exists(),
                str(self.launch_vllm)
                if self.launch_vllm.exists()
                else f"{self.launch_vllm} missing. Run `autodspark setup`.",
            )
        )
        return rows


def _module_version(python: Path, module: str) -> str | None:
    code = (
        "import importlib.metadata as m, importlib, sys;"
        f"importlib.import_module({module.replace('-', '_')!r});"
        f"print(m.version({module.replace('_', '-')!r}))"
    )
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def find_nvcc() -> str | None:
    """nvcc from PATH, CUDA_HOME/CUDA_PATH or the default toolkit location."""
    candidates = [shutil.which("nvcc")]
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            candidates.append(str(Path(os.environ[var]) / "bin" / "nvcc"))
    candidates.append("/usr/local/cuda/bin/nvcc")
    return next((c for c in candidates if c and Path(c).exists()), None)


def flashinfer_status(rt: Runtime) -> tuple[bool, str]:
    """vLLM's sampler uses FlashInfer kernels; they must be precompiled or compilable."""
    cached = _module_version(rt.vllm_python, "flashinfer_jit_cache")
    if cached:
        return True, f"precompiled (flashinfer-jit-cache {cached})"
    nvcc = find_nvcc()
    if nvcc:
        return True, (
            f"not precompiled; FlashInfer compiles them with {nvcc} on the first start "
            "(a few minutes, cached afterwards)"
        )
    return False, (
        "not precompiled and no nvcc found, so vLLM will fail to start. Run "
        f"`{rt.vllm_python.parent / 'vllm'} download-kernels`, or install the CUDA toolkit."
    )


def setup_commands(
    rt: Runtime, vllm_spec: str | None = None, python_version: str = "3.12"
) -> list[list[str]]:
    """Commands that build both environments and fetch the pinned Speculators scripts."""
    vllm_spec = vllm_spec or f"vllm>={VLLM_MIN_VERSION}"
    spec_env = rt.speculators_python.parent.parent
    vllm_env = rt.vllm_python.parent.parent
    cmds = [
        ["uv", "venv", str(spec_env), "--python", python_version],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(rt.speculators_python),
            f"speculators=={SPECULATORS_VERSION}",
        ],
        ["uv", "venv", str(vllm_env), "--python", python_version],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(rt.vllm_python),
            vllm_spec,
            f"hs-connectors=={SPECULATORS_VERSION}",
        ],
    ]
    return cmds


def speculators_archive_url() -> str:
    return f"{SPECULATORS_ARCHIVE}/{SPECULATORS_GIT_TAG}"


def fetch_speculators_src(rt: Runtime, timeout: float = 300) -> None:
    """Download the pinned Speculators source (for scripts/launch_vllm.py).

    Uses GitHub's tarball endpoint rather than git, so it works without git
    and follows HTTP(S)_PROXY like any other download.
    """
    dest = rt.speculators_src
    tmp = dest.with_name(dest.name + ".download")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tempfile.TemporaryFile() as buf:
        with httpx.stream(
            "GET", speculators_archive_url(), follow_redirects=True, timeout=timeout
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                buf.write(chunk)
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            _safe_extract(tar, tmp, strip=1)
    if not (tmp / "scripts" / "launch_vllm.py").exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("Downloaded Speculators archive has no scripts/launch_vllm.py")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def _safe_extract(tar: tarfile.TarFile, dest: Path, strip: int = 0) -> None:
    """Extract regular files and dirs only, refusing anything outside `dest`."""
    root = dest.resolve()
    for member in tar.getmembers():
        parts = Path(member.name).parts[strip:]
        if not parts or not (member.isfile() or member.isdir()):
            continue
        target = (root / Path(*parts)).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        src = tar.extractfile(member)
        if src is None:
            continue
        with src, target.open("wb") as out:
            shutil.copyfileobj(src, out)
        target.chmod(0o755 if member.mode & 0o111 else 0o644)


def missing_setup_tools() -> list[str]:
    return [tool for tool in ("uv",) if shutil.which(tool) is None]
