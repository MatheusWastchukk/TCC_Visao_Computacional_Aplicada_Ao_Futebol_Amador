"""A escolha explícita exige o fornecedor e a GPU corretos, sem retorno silencioso à CPU."""
from types import SimpleNamespace

import pytest

from app.accelerators import AcceleratorError, runtime_vendor, select_device


def fake_torch(vendor="cpu", available=False, count=0, capability=(8, 6), cuda="12.6"):
    return SimpleNamespace(
        version=SimpleNamespace(hip="7.2" if vendor == "amd" else None,
                                cuda=cuda if vendor == "nvidia" else None),
        cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: count,
                             get_device_capability=lambda index: capability,
                             get_device_name=lambda index: "NVIDIA GeForce RTX 5060 Ti"),
    )


@pytest.mark.parametrize("vendor", ["cpu", "nvidia", "amd"])
def test_cpu_is_explicit_even_with_gpu_runtime(vendor):
    assert select_device("cpu", fake_torch(vendor, True, 1)) == "cpu"


def test_amd_reuses_cuda_api_but_is_not_nvidia():
    runtime = fake_torch("amd", True, 1)
    assert runtime_vendor(runtime) == "amd"
    assert select_device("amd", runtime) == "cuda:0"
    with pytest.raises(AcceleratorError, match="exige o componente"):
        select_device("nvidia", runtime)


@pytest.mark.parametrize("mode", ["nvidia", "amd"])
def test_gpu_choice_fails_without_available_device(mode):
    with pytest.raises(AcceleratorError) as caught:
        select_device(mode, fake_torch(mode, False, 0))
    assert caught.value.code == "GPU_UNAVAILABLE"


def test_gpu_index_must_still_exist():
    assert select_device("nvidia", fake_torch("nvidia", True, 2), 1) == "cuda:1"
    with pytest.raises(AcceleratorError) as caught:
        select_device("nvidia", fake_torch("nvidia", True, 1), 1)
    assert caught.value.code == "GPU_INDEX_UNAVAILABLE"


def test_blackwell_rejects_old_cuda_before_running_kernels():
    runtime = fake_torch("nvidia", True, 1, capability=(12, 0))
    with pytest.raises(AcceleratorError) as caught:
        select_device("nvidia", runtime)
    assert caught.value.code == "CUDA_ARCHITECTURE_UNSUPPORTED"
    assert "RTX 5060 Ti" in str(caught.value)
    assert "CUDA 12.8" in str(caught.value)


@pytest.mark.parametrize("cuda", ["12.8", "12.9", "13.0"])
def test_blackwell_can_proceed_to_execution_test_with_new_cuda(cuda):
    runtime = fake_torch("nvidia", True, 1, capability=(12, 0), cuda=cuda)
    assert select_device("nvidia", runtime) == "cuda:0"


def test_amd_does_not_require_nvidia_capability_api():
    runtime = fake_torch("amd", True, 1)
    del runtime.cuda.get_device_capability
    assert select_device("amd", runtime) == "cuda:0"
