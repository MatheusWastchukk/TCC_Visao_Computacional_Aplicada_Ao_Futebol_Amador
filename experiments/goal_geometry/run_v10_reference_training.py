"""Executa o primeiro treinamento de referência da V10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa o treinamento de referência V10.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    args.project.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        project=str(args.project),
        name="yolo11n_cpu_reference",
        exist_ok=True,
        verbose=True,
    )
    metrics = getattr(results, "results_dict", {})
    summary = {
        "experiment_id": "EXP-0089",
        "version": "V10_reference_training",
        "model": args.model,
        "device": "cpu",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": 0,
        "data": str(args.data),
        "results": {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "status": "reference_training_completed",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
