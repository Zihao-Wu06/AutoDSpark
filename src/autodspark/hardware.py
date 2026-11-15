"""Detect GPUs, disk space and the tools AutoDSpark shells out to."""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path

from pydantic import BaseModel

NETWORK_FILESYSTEMS = {"nfs", "nfs4", "smbfs", "cifs", "afpfs", "fuse.sshfs", "9p", "webdav"}


class GPU(BaseModel):
    index: int
    name: str
    memory_gb: float
    compute_cap: str | None = None
    memory_used_gb: float = 0.0  # held by other processes when we looked
    uuid: str | None = None
    # How CUDA_VISIBLE_DEVICES named this GPU when the user restricted it (e.g. a
    # GPU-... UUID). Subprocesses get this back so we never widen what was allowed.
    cuda_id: str | None = None

    @property
    def free_gb(self) -> float:
        return max(0.0, self.memory_gb - self.memory_used_gb)

    @property
    def visible_id(self) -> str:
        """The value to put in CUDA_VISIBLE_DEVICES for this GPU."""
        return self.cuda_id or str(self.index)


class GPUConfigError(ValueError):
    """GPUs exist, but the environment doesn't let us plan for them."""


class Hardware(BaseModel):
    gpus: list[GPU] = []
    driver_version: str | None = None
    disk_free_gb: float = 0.0
    workdir_fs: str | None = None
    platform: str = ""
    simulated: bool = False
    # Why no GPU is usable although nvidia-smi found some (MIG, CUDA_VISIBLE_DEVICES="").
    gpu_problem: str | None = None

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    @property
    def min_gpu_gb(self) -> float:
        return min((g.memory_gb for g in self.gpus), default=0.0)

    @property
    def workdir_on_network_fs(self) -> bool:
        return (self.workdir_fs or "").lower() in NETWORK_FILESYSTEMS


GPU_QUERY = "index,name,memory.total,compute_cap,memory.used,uuid,mig.mode.current"
GPU_QUERY_BASIC = "index,name,memory.total,compute_cap"  # older drivers


def _field(parts: list[str], i: int) -> str | None:
    if len(parts) <= i or parts[i] in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    return parts[i]


def parse_nvidia_smi(csv_text: str) -> list[GPU]:
    """Parse nvidia-smi output.

    Expects `--query-gpu=<GPU_QUERY> --format=csv,noheader,nounits`; output with
    only the first four columns (GPU_QUERY_BASIC) also parses.
    """
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            mem_mib = float(parts[2])
        except ValueError:
            continue
        try:
            used_mib = float(_field(parts, 4) or 0)
        except ValueError:
            used_mib = 0.0
        gpus.append(
            GPU(
                index=int(parts[0]),
                name=parts[1],
                memory_gb=round(mem_mib / 1024, 1),
                compute_cap=_field(parts, 3),
                memory_used_gb=round(used_mib / 1024, 1),
                uuid=_field(parts, 5),
            )
        )
    return gpus


def mig_enabled_indices(csv_text: str) -> list[int]:
    """GPUs whose `mig.mode.current` column (the 7th of GPU_QUERY) says Enabled."""
    out = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit() and (_field(parts, 6) or "").lower() == "enabled":
            out.append(int(parts[0]))
    return out


MIG_UNSUPPORTED = (
    "MIG is enabled on GPU {which}. AutoDSpark doesn't support MIG slices yet. Disable MIG "
    "(nvidia-smi -mig 0) or point CUDA_VISIBLE_DEVICES at full GPUs."
)


def _match(token: str, gpus: list[GPU]) -> GPU | None:
    if token.isdigit():
        return next((g for g in gpus if g.index == int(token)), None)
    if token.upper().startswith("GPU-"):
        # CUDA accepts any unique prefix of a UUID.
        hits = [g for g in gpus if g.uuid and g.uuid.lower().startswith(token.lower())]
        return hits[0] if len(hits) == 1 else None
    return None


def _visible(gpus: list[GPU], env: str | None = None, mig: list[int] | None = None) -> list[GPU]:
    """The GPUs CUDA_VISIBLE_DEVICES lets us use, in the order it lists them.

    Accepts indices and GPU-... UUIDs (or unique UUID prefixes). Like CUDA, an
    empty value or an entry that names no GPU (e.g. -1) ends the list, so "" and
    "-1" mean no GPUs. Raises GPUConfigError when nothing usable is left or MIG
    is in play.
    """
    if env is None:
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None:
        chosen = list(gpus)
    else:
        chosen, seen = [], set()
        for token in (t.strip() for t in env.split(",")):
            if token.upper().startswith("MIG-"):
                raise GPUConfigError(MIG_UNSUPPORTED.format(which=f"slice {token}"))
            g = _match(token, gpus)
            if g is None:
                break
            if g.index not in seen:
                seen.add(g.index)
                chosen.append(
                    g.model_copy(update={"cuda_id": None if token == str(g.index) else token})
                )
        if gpus and not chosen:
            raise GPUConfigError(
                f"CUDA_VISIBLE_DEVICES={env!r} leaves no GPU to use (it is empty, -1, or names "
                "no GPU nvidia-smi reports). Unset it, or list the GPUs to use by index or "
                "UUID, e.g. CUDA_VISIBLE_DEVICES=0,1."
            )
    mig_on = [g for g in chosen if g.index in set(mig or [])]
    if mig_on:
        raise GPUConfigError(MIG_UNSUPPORTED.format(which=", ".join(str(g.index) for g in mig_on)))
    return chosen


def _smi(args: list[str]) -> str:
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=30, check=True
    ).stdout


def query_gpus() -> tuple[list[GPU], str | None]:
    """Visible GPUs and the driver version. Raises GPUConfigError (see `_visible`)."""
    if not shutil.which("nvidia-smi"):
        return [], None
    try:
        try:
            out = _smi([f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"])
        except subprocess.CalledProcessError:
            out = _smi([f"--query-gpu={GPU_QUERY_BASIC}", "--format=csv,noheader,nounits"])
        try:
            driver = _smi(["--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()
        except subprocess.CalledProcessError:
            driver = []
    except (subprocess.SubprocessError, OSError):
        return [], None
    gpus = _visible(parse_nvidia_smi(out), mig=mig_enabled_indices(out))
    return gpus, (driver[0].strip() if driver else None)


def _unescape_mount(text: str) -> str:
    """/proc/mounts writes spaces and tabs in mount points as octal escapes (\\040)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), text)


def _is_under(path: str, mount_point: str) -> bool:
    """True if `path` is `mount_point` or inside it, comparing whole path components."""
    mount_point = mount_point.rstrip("/") or "/"
    return mount_point == "/" or path == mount_point or path.startswith(mount_point + "/")


def best_mount(path: str, mounts: list[tuple[str, str]]) -> str | None:
    """Filesystem type of the deepest (mount_point, fstype) entry holding `path`."""
    best, fstype = "", None
    for mount_point, fs in mounts:
        if _is_under(path, mount_point) and len(mount_point) > len(best):
            best, fstype = mount_point, fs
    return fstype


def filesystem_type(path: Path) -> str | None:
    """Best-effort filesystem type of the mount holding `path`."""
    path = path.resolve()
    mounts: list[tuple[str, str]] = []
    try:
        if platform.system() == "Linux":
            with open("/proc/mounts", encoding="utf-8") as f:
                for line in f:
                    fields = line.split()
                    if len(fields) >= 3:
                        mounts.append((_unescape_mount(fields[1]), fields[2]))
        else:
            out = subprocess.run(["mount"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                if " on " not in line or "(" not in line:
                    continue
                mount_point = line.split(" on ", 1)[1].split(" (", 1)[0]
                mounts.append((mount_point, line.rsplit("(", 1)[1].split(",")[0].strip()))
    except (OSError, subprocess.SubprocessError):
        return None
    return best_mount(str(path), mounts)


def detect(workdir: Path) -> Hardware:
    workdir.mkdir(parents=True, exist_ok=True)
    problem = None
    try:
        gpus, driver = query_gpus()
    except GPUConfigError as e:
        gpus, driver, problem = [], None, str(e)
    return Hardware(
        gpus=gpus,
        driver_version=driver,
        gpu_problem=problem,
        disk_free_gb=round(shutil.disk_usage(workdir).free / 1024**3, 1),
        workdir_fs=filesystem_type(workdir),
        platform=f"{platform.system()} {platform.machine()}",
    )


def simulated(spec: str, disk_free_gb: float = 2000.0) -> Hardware:
    """Fake hardware for `plan --gpus 2x80`-style what-if planning on any machine.

    `spec` is "<count>x<GB>", e.g. "1x24" or "8x80".
    """
    try:
        count_s, mem_s = spec.lower().split("x", 1)
        count, mem = int(count_s), float(mem_s)
    except ValueError as e:
        raise ValueError(f"--gpus must look like 1x24 or 8x80, got {spec!r}") from e
    if count < 1 or mem <= 0:
        raise ValueError(f"--gpus must have at least one GPU with positive memory, got {spec!r}")
    return Hardware(
        gpus=[GPU(index=i, name=f"simulated-{mem:g}GB", memory_gb=mem) for i in range(count)],
        disk_free_gb=disk_free_gb,
        platform="simulated",
        simulated=True,
    )


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True
