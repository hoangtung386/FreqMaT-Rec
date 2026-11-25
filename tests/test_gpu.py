import subprocess

from freqmat import gpu


def test_prefers_gpu_1_then_0_then_first(monkeypatch):
    monkeypatch.delenv(gpu.ENV_OVERRIDE, raising=False)
    assert gpu.choose_gpu(["0", "1"]) == "1"
    assert gpu.choose_gpu(["0", "1", "2", "3"]) == "1"
    assert gpu.choose_gpu(["0"]) == "0"
    assert gpu.choose_gpu(["2", "3"]) == "2"
    assert gpu.choose_gpu([]) is None


def test_environment_override_wins(monkeypatch):
    monkeypatch.setenv(gpu.ENV_OVERRIDE, "0")
    assert gpu.choose_gpu(["0", "1"]) == "0"


def test_physical_gpus_reads_nvidia_smi_and_respects_visible_devices(monkeypatch):
    out = subprocess.CompletedProcess([], 0, stdout="0\n1\n2\n")
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: out)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu.physical_gpus() == ["0", "1", "2"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    assert gpu.physical_gpus() == ["2", "0"]


def test_no_nvidia_smi_means_no_gpu(monkeypatch):
    def missing(*_a, **_k):
        raise FileNotFoundError

    monkeypatch.setattr(gpu.subprocess, "run", missing)
    assert gpu.physical_gpus() == []
    assert gpu.choose_gpu() is None
