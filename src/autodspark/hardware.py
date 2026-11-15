"""Detect GPUs, disk space and the tools AutoDSpark shells out to."""

from __future__ import annotations

import os
import platform
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


class Hardware(BaseModel):
    gpus: list[GPU] = []
    driver_version: str | None = None
    disk_free_gb: float = 0.0
    workdir_fs: str | None = None
    platform: str = ""
    simulated: bool = False

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    @property
    def min_gpu_gb(self) -> float:
        return min((g.memory_gb for g in self.gpus), default=0.0)

    @property
    def workdir_on_network_fs(self) -> bool:
        return (self.workdir_fs or "").lower() in NETWORK_FILESYSTEMS


def parse_nvidia_smi(csv_text: str) -> list[GPU]:
    """Parse nvidia-smi output.

    Expects `--query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits`.
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
        gpus.append(
            GPU(
                index=int(parts[0]),
                name=parts[1],
                memory_gb=round(mem_mib / 1024, 1),
                compute_cap=parts[3] if len(parts) > 3 and parts[3] not in ("", "[N/A]") else None,
            )
        )
    return gpus


def _visible(gpus: list[GPU]) -> list[GPU]:
    """Respect CUDA_VISIBLE_DEVICES when it lists plain indices."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not env:
        return gpus
    wanted = [s.strip() for s in env.split(",") if s.strip()]
    if not all(w.isdigit() for w in wanted):
        return gpus
    by_index = {g.index: g for g in gpus}
    return [by_index[int(w)] for w in wanted if int(w) in by_index]


def query_gpus() -> tuple[list[GPU], str | None]:
    if not shutil.which("nvidia-smi"):
        return [], None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            .stdout.strip()
            .splitlines()
        )
    except (subprocess.SubprocessError, OSError):
        return [], None
    return _visible(parse_nvidia_smi(out)), (driver[0] if driver else None)


def filesystem_type(path: Path) -> str | None:
    """Best-effort filesystem type of the mount holding `path`."""
    path = path.resolve()
    try:
        if platform.system() == "Linux":
            best, fstype = "", None
            with open("/proc/mounts", encoding="utf-8") as f:
                for line in f:
                    fields = line.split()
                    if (
                        len(fields) >= 3
                        and str(path).startswith(fields[1])
                        and len(fields[1]) > len(best)
                    ):
                        best, fstype = fields[1], fields[2]
            return fstype
        out = subprocess.run(["mount"], capture_output=True, text=True, timeout=10).stdout
        best, fstype = "", None
        for line in out.splitlines():
            if " on " not in line or "(" not in line:
                continue
            mount_point = line.split(" on ", 1)[1].split(" (", 1)[0]
            if str(path).startswith(mount_point) and len(mount_point) > len(best):
                best = mount_point
                fstype = line.rsplit("(", 1)[1].split(",")[0].strip()
        return fstype
    except (OSError, subprocess.SubprocessError):
        return None


def detect(workdir: Path) -> Hardware:
    workdir.mkdir(parents=True, exist_ok=True)
    gpus, driver = query_gpus()
    return Hardware(
        gpus=gpus,
        driver_version=driver,
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
