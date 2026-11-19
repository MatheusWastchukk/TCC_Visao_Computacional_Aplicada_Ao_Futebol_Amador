"""Valida o fornecedor do runtime e operações reais antes de autorizar uma GPU."""
from __future__ import annotations

from dataclasses import dataclass


MODES = {"cpu": "CPU", "nvidia": "GPU NVIDIA", "amd": "GPU AMD"}


@dataclass
class AcceleratorError(RuntimeError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def runtime_vendor(torch_module) -> str:
    """ROCm também usa torch.cuda; torch.version distingue os fornecedores."""
    if getattr(torch_module.version, "hip", None):
        return "amd"
    if getattr(torch_module.version, "cuda", None):
        return "nvidia"
    return "cpu"


def select_device(mode: str, torch_module=None, index: int | None = None) -> str:
    """Confere runtime e dispositivo, sem substituir uma GPU solicitada por CPU."""
    if mode not in MODES:
        raise AcceleratorError("INVALID_MODE", "Escolha CPU, GPU NVIDIA ou GPU AMD.")
    if torch_module is None:
        import torch as torch_module
    if mode == "cpu":
        return "cpu"
    vendor = runtime_vendor(torch_module)
    if vendor != mode:
        raise AcceleratorError(
            "RUNTIME_MISMATCH",
            f"O componente carregado é {MODES[vendor]}; a opção {MODES[mode]} exige o componente correspondente.",
        )
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise AcceleratorError(
            "GPU_UNAVAILABLE",
            f"Nenhuma {MODES[mode]} compatível ficou disponível para processamento. "
            "Verifique a placa e o driver ou escolha CPU.",
        )
    selected = 0 if index is None else index
    if selected < 0 or selected >= torch_module.cuda.device_count():
        raise AcceleratorError("GPU_INDEX_UNAVAILABLE", "A GPU selecionada não está mais disponível. Abra o aplicativo novamente.")
    if mode == "nvidia":
        capability = torch_module.cuda.get_device_capability(selected)
        cuda_version = tuple(int(part) for part in torch_module.version.cuda.split(".")[:2])
        # As GPUs Blackwell de consumo exigem o pacote com suporte a sm_120.
        if capability == (12, 0) and cuda_version < (12, 8):
            name = torch_module.cuda.get_device_name(selected)
            raise AcceleratorError(
                "CUDA_ARCHITECTURE_UNSUPPORTED",
                f"A {name} foi detectada, mas o componente NVIDIA deste aplicativo usa "
                f"CUDA {torch_module.version.cuda} e não oferece suporte a essa arquitetura "
                "(Blackwell, capacidade 12.0). Atualize o aplicativo e reinstale o componente "
                "NVIDIA com CUDA 12.8 ou superior. Até lá, escolha CPU.",
            )
    return f"cuda:{selected}"


def exercise_device(mode: str, *, index: int | None = None) -> dict[str, object]:
    """Exercita alocação, multiplicação, convolução e NMS no dispositivo escolhido."""
    import torch
    from torchvision.ops import nms

    device = select_device(mode, torch, index)
    try:
        with torch.inference_mode():
            matrix = torch.ones((16, 16), device=device)
            product = matrix @ matrix
            convolution = torch.nn.Conv2d(3, 4, 3).to(device)
            output = convolution(torch.ones((1, 3, 32, 32), device=device))
            boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]], device=device)
            keep = nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
            if mode != "cpu":
                torch.cuda.synchronize(int(device.split(":")[1]))
            if not torch.isfinite(output).all().item() or product[0, 0].item() != 16 or not len(keep):
                raise RuntimeError("O teste numérico não produziu o resultado esperado.")
    except Exception as error:
        raise AcceleratorError(
            "DEVICE_EXECUTION_FAILED",
            f"A opção {MODES[mode]} foi detectada, mas falhou no teste de execução. "
            f"Verifique a compatibilidade da placa, do driver e do componente. Detalhe: {error}",
        ) from error
    return {
        "mode": mode, "device": device, "runtime_vendor": runtime_vendor(torch),
        "device_name": "CPU" if mode == "cpu" else torch.cuda.get_device_name(int(device.split(":")[1])),
        "torch": torch.__version__, "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None),
    }
