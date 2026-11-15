"""Where the heavy environments live, and how to create them.

Speculators and vLLM are installed into two separate virtualenvs, as the
Speculators docs recommend, so their torch pins never conflict. AutoDSpark
itself stays a light CLI that drives both.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx
from pydantic import BaseModel

from . import SPECULATORS_GIT_TAG, SPECULATORS_VERSION, VLLM_VERSION

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

    @property
    def speculators_src_managed(self) -> bool:
        """True when the Speculators source is AutoDSpark's own copy under `home`.

        Anything else came from the user (AUTODSPARK_SPECULATORS_SRC or an argument)
        and must never be deleted or replaced.
        """
        return self.speculators_src.resolve() == (self.home / "speculators-src").resolve()

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
            if version is None:
                rows.append((label, False, f"`import {mod}` failed in {py}"))
            elif mod == "vllm" and version.split("+")[0] != VLLM_VERSION:
                # A warning, not a failure: newer vLLM often works, it just isn't tested.
                rows.append(
                    (
                        label,
                        True,
                        f"{version} (warning: AutoDSpark is tested with vLLM {VLLM_VERSION}; "
                        "re-run `autodspark setup` to install it)",
                    )
                )
            else:
                rows.append((label, True, version))
        if self.vllm_python.exists():
            hs = _module_version(self.vllm_python, "hs_connectors")
            rows.append(
                (
                    "hs_connectors in vLLM env",
                    hs is not None,
                    hs or "missing; install hs-connectors into the vLLM env",
                )
            )
            rows.append(("CUDA in vLLM env", *cuda_status(self.vllm_python)))
        if self.launch_vllm.exists():
            scripts_detail = str(self.launch_vllm)
        elif self.speculators_src_managed:
            scripts_detail = f"{self.launch_vllm} missing. Run `autodspark setup`."
        else:
            scripts_detail = (
                f"{self.launch_vllm} missing. Point AUTODSPARK_SPECULATORS_SRC at a Speculators "
                f"{SPECULATORS_GIT_TAG} source tree (setup only downloads into it if it is "
                "empty), or unset it."
            )
        rows.append(("Speculators scripts", self.launch_vllm.exists(), scripts_detail))
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


_CUDA_PROBE = """
import json, torch
info = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": False}
if torch.cuda.is_available():
    info["available"] = True
    info["devices"] = [
        [torch.cuda.get_device_name(i), *torch.cuda.get_device_capability(i)]
        for i in range(torch.cuda.device_count())
    ]
print(json.dumps(info))
"""


def cuda_status(python: Path, timeout: float = 60) -> tuple[bool, str]:
    """Ask the env's torch whether it can use the GPU, and which compute capability it sees.

    Catches the common case of a driver too old for the CUDA build of torch, which
    otherwise only shows up when vLLM fails to start.
    """
    try:
        out = subprocess.run(
            [str(python), "-c", _CUDA_PROBE], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return True, f"warning: not checked, torch took over {timeout:g} s to start"
    except (OSError, subprocess.SubprocessError) as e:
        return True, f"warning: not checked ({e})"
    lines = out.stdout.strip().splitlines()
    try:
        info = json.loads(lines[-1]) if out.returncode == 0 and lines else None
    except ValueError:
        info = None
    if not isinstance(info, dict):
        err = (out.stderr.strip().splitlines() or ["no output"])[-1]
        return False, f"`import torch` failed in {python}: {err}"
    build = f"torch {info.get('torch')}"
    if not info.get("cuda"):
        return False, f"{build} is a CPU-only build; the vLLM env needs a CUDA build of torch"
    build += f" (CUDA {info['cuda']})"
    if not info.get("available"):
        hint = _stderr_hint(out.stderr)
        return False, (
            f"{build} cannot use the GPU"
            + (f": {hint[:200]}" if hint else "")
            + ". Usually the NVIDIA driver is too old for this torch build; update the driver "
            "(`nvidia-smi` shows the newest CUDA version it supports)."
        )
    groups: dict[str, int] = {}
    for name, major, minor in info.get("devices") or []:
        key = f"{name} (compute {major}.{minor})"
        groups[key] = groups.get(key, 0) + 1
    gpus = ", ".join(f"{n}x {k}" if n > 1 else k for k, n in groups.items())
    return True, f"{build}: {gpus or 'GPU available'}"


def _stderr_hint(stderr: str) -> str:
    """The most telling stderr line: the last warning/error message, else the last line."""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "Warning:" in ln or "Error:" in ln:
            return ln.split("Warning:", 1)[-1].split("Error:", 1)[-1].strip()
    return lines[-1] if lines else ""


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
    """Commands that build both environments and fetch the pinned Speculators scripts.

    Safe to re-run (e.g. after a network failure): an existing virtualenv is reused rather
    than recreated, since `uv venv` refuses to overwrite one without `--clear`.
    """
    vllm_spec = vllm_spec or f"vllm=={VLLM_VERSION}"
    spec_env = rt.speculators_python.parent.parent
    vllm_env = rt.vllm_python.parent.parent

    def venv(env: Path) -> list[list[str]]:
        if (env / "pyvenv.cfg").exists() and (env / "bin" / "python").exists():
            return []
        create = ["uv", "venv", str(env), "--python", python_version]
        if env.exists():  # half-built or broken env from an earlier attempt: rebuild it
            return [["rm", "-rf", str(env)], create]
        return [create]

    cmds = [
        *venv(spec_env),
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(rt.speculators_python),
            f"speculators=={SPECULATORS_VERSION}",
        ],
        *venv(vllm_env),
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

    Only AutoDSpark's own copy under `home` is ever replaced. A user-supplied path
    (AUTODSPARK_SPECULATORS_SRC) is filled only if it is missing or empty.
    """
    dest = rt.speculators_src
    managed = rt.speculators_src_managed
    if not managed and (dest.is_file() or (dest.is_dir() and any(dest.iterdir()))):
        raise RuntimeError(
            f"{dest} (AUTODSPARK_SPECULATORS_SRC) is not empty but has no "
            f"scripts/launch_vllm.py, so it is not a Speculators source tree. Point it at a "
            f"Speculators {SPECULATORS_GIT_TAG} checkout, or unset it to let setup download "
            "one. AutoDSpark never deletes a folder you chose."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.download-", dir=dest.parent))
    try:
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
            raise RuntimeError("Downloaded Speculators archive has no scripts/launch_vllm.py")
        if managed:
            shutil.rmtree(dest, ignore_errors=True)
        elif dest.is_dir():
            dest.rmdir()  # empty, checked above; rmdir refuses anything else
        tmp.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
