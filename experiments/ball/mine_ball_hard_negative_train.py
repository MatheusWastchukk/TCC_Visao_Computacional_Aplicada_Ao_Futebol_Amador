"""Localiza falsos positivos de alta confiança nas imagens negativas do treino."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Minera negativos difíceis para o detector da bola.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["split"] == "train" and row["ball_visibility"] == "NOT_VISIBLE"
        ]
    if not rows:
        raise ValueError("Não há negativos NOT_VISIBLE no split de treino.")
    model = YOLO(str(args.model))
    mined: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        result = model.predict(
            source=image,
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        confidences = [float(value) for value in result.boxes.conf.tolist()]
        if confidences:
            mined.append({
                **row,
                "false_prediction_count": str(len(confidences)),
                "top_false_confidence": f"{max(confidences):.6f}",
            })
        if index % 50 == 0 or index == len(rows):
            print(f"Processados: {index}/{len(rows)}; negativos_dificeis={len(mined)}")
    mined.sort(key=lambda row: float(row["top_false_confidence"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ["false_prediction_count", "top_false_confidence"]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mined)
    print(f"negativos_treino={len(rows)}")
    print(f"negativos_dificeis={len(mined)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
