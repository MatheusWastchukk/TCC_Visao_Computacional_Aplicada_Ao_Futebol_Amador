"""Gera uma evidência visual de uma previsão individual do detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera overlay de uma previsão V10.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    model = YOLO(str(args.model))
    result = model.predict(str(args.image), imgsz=640, conf=args.confidence, device="cpu", verbose=False)[0]
    image = result.plot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise ValueError(f"Não foi possível salvar o overlay: {args.output}")
    print(f"deteccoes={len(result.boxes)}")
    for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True):
        print(f"box={box}; confidence={confidence:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
