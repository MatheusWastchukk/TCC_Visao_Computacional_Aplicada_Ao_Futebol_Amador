"""Valida a inferência ONNX usando o provedor DirectML da GPU AMD."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser(description="Testa ONNX Runtime DirectML com uma imagem.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.model), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {args.image}")
    image = cv2.cvtColor(cv2.resize(image, (640, 640)), cv2.COLOR_BGR2RGB)
    tensor = image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    output = session.run(None, {session.get_inputs()[0].name: tensor})
    print(f"provedores={session.get_providers()}")
    print(f"status=ok shape_saida={output[0].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
