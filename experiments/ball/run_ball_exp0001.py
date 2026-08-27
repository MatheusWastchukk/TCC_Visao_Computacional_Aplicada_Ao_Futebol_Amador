"""Executa o baseline inicial do detector da bola."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina o baseline BALL-EXP-0001.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        verbose=False,
    )
    metrics = dict(results.results_dict or {})
    summary = {
        "experiment_id": "BALL-EXP-0001",
        "model": str(args.model),
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metrics": {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "execution_seconds": round(time.perf_counter() - started, 3),
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
