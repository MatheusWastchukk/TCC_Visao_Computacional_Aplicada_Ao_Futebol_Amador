"""Seleciona exemplos positivos difíceis no split de treino."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Encontra exemplos difíceis no treino da bola.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "train" and row["ball_visibility"] in {"VISIBLE", "PARTIAL"}]
    model = YOLO(str(args.model))
    hard: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        result = model.predict(source=image, conf=args.conf, iou=0.7, imgsz=960, verbose=False, device=args.device)[0]
        gt = tuple(float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"))
        best_iou = 0.0
        best_conf = 0.0
        for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            overlap = iou(gt, tuple(float(value) for value in box))
            if overlap > best_iou:
                best_iou = overlap
                best_conf = float(confidence)
        if best_iou < 0.5:
            hard.append({**row, "mined_iou": f"{best_iou:.6f}", "mined_confidence": f"{best_conf:.6f}"})
        if index % 50 == 0 or index == len(rows):
            print(f"Processados: {index}/{len(rows)}; difíceis={len(hard)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ["mined_iou", "mined_confidence"]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hard)
    print(f"positivos_treino={len(rows)}")
    print(f"dificeis={len(hard)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
