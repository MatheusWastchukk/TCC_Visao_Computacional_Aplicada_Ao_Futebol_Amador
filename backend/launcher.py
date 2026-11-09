"""Ponto de entrada único para desenvolvimento e backend empacotado em CPU."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
from pathlib import Path

from app.version import VERSION

# O aplicativo usa apenas dependências instaladas no build; nunca executa pip.
os.environ["YOLO_AUTOINSTALL"] = "false"


def self_check() -> dict[str, object]:
    """Exercita importações, modelos e binários sem rede nem vídeos de avaliação."""
    import joblib
    import numpy as np
    import torch
    from torchvision.models import resnet18
    from ultralytics import YOLO

    from app.runtime import encoder_weights_path, ffmpeg_path, processing_device, resource_root
    from backend.server import app, device_metrics, model_paths

    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("O pacote CPU exige PyTorch sem runtime de GPU.")
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
        "status": "ok", "version": VERSION, "runtime_vendor": "cpu", "device": "cpu", "torch": torch.__version__,
        "models_loaded": list(model_paths()), "resnet18": "ok",
        "ffmpeg": result.stdout.splitlines()[0], "api": app.title, "metrics": device_metrics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicia a API local ou verifica o pacote CPU.")
    parser.add_argument("--self-check", action="store_true", help="Verifica o pacote sem iniciar o servidor.")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False), flush=True)
        return
    import uvicorn

    from backend.server import app

    from app.runtime import processing_device

    processing_device()

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")



if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
