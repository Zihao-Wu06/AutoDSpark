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
