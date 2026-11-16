"""Ponto de entrada único para desenvolvimento e backend empacotado em CPU."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
from pathlib import Path

from app.version import VERSION
from app.vendor_runtime import activate_vendor_runtime

activate_vendor_runtime()

# O aplicativo usa apenas dependências instaladas no build; nunca executa pip.
os.environ["YOLO_AUTOINSTALL"] = "false"


def self_check(expected_runtime: str = "cpu") -> dict[str, object]:
    """Exercita importações, modelos e binários sem rede nem vídeos de avaliação."""
    import joblib
    import numpy as np
    import torch
    from torchvision.models import resnet18
    from ultralytics import YOLO

    from app.runtime import encoder_weights_path, ffmpeg_path, processing_device, resource_root
    from backend.server import app, device_metrics, model_paths

    from app.accelerators import runtime_vendor

    if runtime_vendor(torch) != expected_runtime:
        raise RuntimeError(f"O componente exige o runtime {expected_runtime}, mas foi construído com {runtime_vendor(torch)}.")
    root = resource_root()
    if not (root / "experiments/ball/mine_goal_event_candidates_v1.py").is_file():
        raise FileNotFoundError("Estágio de mineração ausente do pacote.")
    encoder_weights = encoder_weights_path()
    if encoder_weights is None or not encoder_weights.is_file():
        raise FileNotFoundError("Pesos locais do ResNet18 ausentes.")
    torch.set_num_threads(2)
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(encoder_weights, map_location="cpu", weights_only=True))
    encoder.eval()
    with torch.inference_mode():
        if encoder(torch.zeros(1, 3, 224, 224)).shape != (1, 1000):
            raise RuntimeError("Saída inesperada do ResNet18.")
    for name, model_path in model_paths().items():
        if name.endswith("detector"):
            YOLO(str(model_path)).predict(np.zeros((320, 320, 3), dtype=np.uint8), device="cpu", imgsz=320, verbose=False)
        else:
            joblib.load(model_path)
    result = subprocess.run([str(ffmpeg_path()), "-version"], capture_output=True, text=True, check=True)
    return {
        "status": "ok", "version": VERSION, "runtime_vendor": runtime_vendor(torch), "device": "cpu", "torch": torch.__version__,
        "models_loaded": list(model_paths()), "resnet18": "ok",
        "ffmpeg": result.stdout.splitlines()[0], "api": app.title, "metrics": device_metrics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicia a API local ou verifica o pacote CPU.")
    parser.add_argument("--self-check", action="store_true", help="Verifica o pacote sem iniciar o servidor.")
    parser.add_argument("--expected-runtime", choices=("cpu", "nvidia", "amd"), default="cpu")
    parser.add_argument("--probe-device", choices=("cpu", "nvidia", "amd"), help="Valida a opção de processamento sem iniciar a API.")
    parser.add_argument("--probe-output", type=Path, help="Arquivo JSON com o resultado da validação.")
    parser.add_argument("--install-runtime", choices=('nvidia', 'amd'), help="Prepara pacotes oficiais já baixados pelo instalador.")
    parser.add_argument("--runtime-downloads", type=Path)
    parser.add_argument("--gpu-root", type=Path)
    parser.add_argument("--gpu-catalog", type=Path)
    parser.add_argument("--remove-runtimes", action='store_true')
    args = parser.parse_args()
    if args.install_runtime or args.remove_runtimes:
        from app.gpu_packages import install_runtime, remove_runtimes
        if args.gpu_root is None:
            parser.error('Informe --gpu-root.')
        if args.remove_runtimes:
            remove_runtimes(args.gpu_root)
        else:
            if args.runtime_downloads is None or args.gpu_catalog is None:
                parser.error('Informe os downloads e o catálogo de GPU.')
            print(json.dumps(install_runtime(args.install_runtime, args.runtime_downloads, args.gpu_root, args.gpu_catalog)))
        return
    if args.probe_device:
        result = probe_device(args.probe_device)
        if args.probe_output:
            args.probe_output.parent.mkdir(parents=True, exist_ok=True)
            args.probe_output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        raise SystemExit(0 if result["ok"] else 2)
    if args.self_check:
        print(json.dumps(self_check(args.expected_runtime), ensure_ascii=False), flush=True)
        return
    import uvicorn

    from backend.server import app

    from app.runtime import processing_device

    processing_device()

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


def probe_device(mode: str) -> dict[str, object]:
    """Valida bibliotecas e modelos no dispositivo real; erros viram mensagens legíveis."""
    try:
        from app.accelerators import AcceleratorError, exercise_device, select_device
        import torch

        select_device(mode, torch)
        import numpy as np
        from ultralytics import YOLO

        from backend.server import model_paths

        failures = []
        for index in ([None] if mode == "cpu" else range(torch.cuda.device_count())):
            try:
                report = exercise_device(mode, index=index)
                for name, model_path in model_paths().items():
                    if not name.endswith("detector"):
                        continue
                    result = YOLO(str(model_path)).predict(
                        np.zeros((320, 320, 3), dtype=np.uint8), device=report["device"], imgsz=320, verbose=False,
                    )[0]
                    if result.boxes.data.device.type != ("cpu" if mode == "cpu" else "cuda"):
                        raise AcceleratorError("DEVICE_FALLBACK", "O detector não executou no dispositivo selecionado.")
                return {"ok": True, "version": VERSION, **report}
            except Exception as error:
                failures.append(str(error))
        raise AcceleratorError("DEVICE_EXECUTION_FAILED", failures[-1])
    except Exception as error:
        return {"ok": False, "version": VERSION, "mode": mode,
                "code": getattr(error, "code", "VALIDATION_FAILED"), "error": str(error)}


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
