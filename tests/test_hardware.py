import pytest

from autodspark import hardware

SMI = """0, NVIDIA A100-SXM4-80GB, 81920, 8.0
1, NVIDIA A100-SXM4-80GB, 81920, 8.0
2, NVIDIA GeForce RTX 4090, 24564, 8.9
"""


def test_parse_nvidia_smi():
    gpus = hardware.parse_nvidia_smi(SMI)
    assert [g.index for g in gpus] == [0, 1, 2]
    assert gpus[0].memory_gb == 80.0
    assert gpus[2].memory_gb == pytest.approx(24.0, abs=0.1)
    assert gpus[2].compute_cap == "8.9"


def test_parse_ignores_junk():
    assert hardware.parse_nvidia_smi("No devices were found\n") == []


def test_cuda_visible_devices(monkeypatch):
    gpus = hardware.parse_nvidia_smi(SMI)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    assert [g.index for g in hardware._visible(gpus)] == [2, 0]


def test_simulated():
    hw = hardware.simulated("4x80")
    assert hw.num_gpus == 4
    assert hw.min_gpu_gb == 80
    assert hw.simulated


@pytest.mark.parametrize("bad", ["80", "0x80", "ax80", "2x0"])
def test_simulated_rejects_bad_spec(bad):
    with pytest.raises(ValueError):
        hardware.simulated(bad)


def test_network_fs_flag():
    hw = hardware.Hardware(workdir_fs="smbfs")
    assert hw.workdir_on_network_fs
    assert not hardware.Hardware(workdir_fs="ext4").workdir_on_network_fs


# index, name, memory.total, compute_cap, memory.used, uuid, mig.mode.current
SMI_FULL = (
    "0, NVIDIA A100-SXM4-80GB, 81920, 8.0, 3, GPU-aaaa1111-0000-0000-0000-000000000000, Disabled\n"
    "1, NVIDIA A100-SXM4-80GB, 81920, 8.0, 40960, GPU-bbbb2222-0000-0000-0000-000000000000, "
    "Disabled\n"
    "2, NVIDIA GeForce RTX 4090, 24564, 8.9, 0, GPU-cccc3333-0000-0000-0000-000000000000, [N/A]\n"
)
UUID_A = "GPU-aaaa1111-0000-0000-0000-000000000000"
UUID_C = "GPU-cccc3333-0000-0000-0000-000000000000"


def test_parse_full_query():
    gpus = hardware.parse_nvidia_smi(SMI_FULL)
    assert gpus[0].uuid == UUID_A
    assert gpus[1].memory_used_gb == 40.0
    assert gpus[1].free_gb == 40.0
    assert gpus[2].memory_used_gb == 0.0
    assert hardware.mig_enabled_indices(SMI_FULL) == []


def test_cuda_visible_devices_uuids(monkeypatch):
    gpus = hardware.parse_nvidia_smi(SMI_FULL)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{UUID_C},GPU-aaaa1111")  # full id and prefix
    vis = hardware._visible(gpus)
    assert [g.index for g in vis] == [2, 0]
    assert [g.visible_id for g in vis] == [UUID_C, "GPU-aaaa1111"]


def test_cuda_visible_devices_indices_keep_plain_ids(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    vis = hardware._visible(hardware.parse_nvidia_smi(SMI_FULL))
    assert [g.visible_id for g in vis] == ["2", "0"]
    assert all(g.cuda_id is None for g in vis)


@pytest.mark.parametrize("value", ["", "-1", " ", "GPU-nomatch", "7"])
def test_cuda_visible_devices_hiding_everything(monkeypatch, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    with pytest.raises(hardware.GPUConfigError, match="no GPU"):
        hardware._visible(hardware.parse_nvidia_smi(SMI_FULL))


def test_cuda_visible_devices_stops_at_invalid_entry(monkeypatch):
    """Like CUDA: devices after an invalid entry (e.g. -1) are not visible."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,-1,2")
    assert [g.index for g in hardware._visible(hardware.parse_nvidia_smi(SMI_FULL))] == [1]


def test_unset_cuda_visible_devices_means_all(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert len(hardware._visible(hardware.parse_nvidia_smi(SMI_FULL))) == 3


def test_mig_uuid_rejected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-1234abcd-0000-0000-0000-000000000000")
    with pytest.raises(hardware.GPUConfigError, match="MIG"):
        hardware._visible(hardware.parse_nvidia_smi(SMI_FULL))


def test_mig_mode_rejected(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    smi = SMI_FULL.replace(
        "40960, GPU-bbbb2222-0000-0000-0000-000000000000, Disabled",
        "40960, GPU-bbbb2222-0000-0000-0000-000000000000, Enabled",
    )
    assert hardware.mig_enabled_indices(smi) == [1]
    with pytest.raises(hardware.GPUConfigError, match="MIG is enabled on GPU 1"):
        hardware._visible(hardware.parse_nvidia_smi(smi), mig=[1])
    # A MIG GPU the user excluded doesn't matter.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert [g.index for g in hardware._visible(hardware.parse_nvidia_smi(smi), mig=[1])] == [0]


def _fake_smi(monkeypatch, out: str, *, full_supported: bool = True):
    import subprocess

    def smi(args):
        if args[0] == f"--query-gpu={hardware.GPU_QUERY}" and not full_supported:
            raise subprocess.CalledProcessError(6, "nvidia-smi")
        if args[0].startswith("--query-gpu=driver_version"):
            return "560.35\n"
        return out

    monkeypatch.setattr(hardware.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(hardware, "_smi", smi)


def test_detect_reports_gpu_problem(monkeypatch, tmp_path):
    _fake_smi(monkeypatch, SMI_FULL)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    hw = hardware.detect(tmp_path)
    assert hw.gpus == []
    assert "CUDA_VISIBLE_DEVICES" in hw.gpu_problem


def test_detect_with_uuids(monkeypatch, tmp_path):
    _fake_smi(monkeypatch, SMI_FULL)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID_C)
    hw = hardware.detect(tmp_path)
    assert [(g.index, g.cuda_id) for g in hw.gpus] == [(2, UUID_C)]
    assert hw.driver_version == "560.35"
    assert hardware.Hardware.model_validate(hw.model_dump(mode="json")) == hw


def test_query_falls_back_on_old_drivers(monkeypatch):
    _fake_smi(monkeypatch, SMI, full_supported=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus, driver = hardware.query_gpus()
    assert [g.index for g in gpus] == [0, 1, 2]
    assert gpus[0].uuid is None and gpus[0].memory_used_gb == 0.0


def test_old_hardware_yaml_still_loads():
    hw = hardware.Hardware.model_validate(
        {"gpus": [{"index": 0, "name": "A100", "memory_gb": 80.0, "compute_cap": "8.0"}]}
    )
    assert hw.gpus[0].memory_used_gb == 0.0 and hw.gpus[0].cuda_id is None
    assert hw.gpu_problem is None


def test_mount_match_uses_path_components():
    mounts = [("/", "ext4"), ("/data", "nfs4"), ("/data2/x", "xfs")]
    assert hardware.best_mount("/data2", mounts) == "ext4"
    assert hardware.best_mount("/data2/x/run", mounts) == "xfs"
    assert hardware.best_mount("/data", mounts) == "nfs4"
    assert hardware.best_mount("/data/run", mounts) == "nfs4"
    assert hardware.best_mount("/data/", [("/data/", "nfs")]) == "nfs"


def test_proc_mounts_escapes():
    assert hardware._unescape_mount("/mnt/my\\040disk") == "/mnt/my disk"
