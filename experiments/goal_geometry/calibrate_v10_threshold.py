"""Calibra o limiar de confianca usando somente as validacoes internas da V10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra o threshold V10 nas validacoes internas dos folds."
    )
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--confidences",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.25],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def experiment_name(path: Path) -> str:
    return next((part for part in path.parts if part.startswith("EXP-")), path.stem)


def main() -> int:
    args = parse_args()
    if len(args.models) != len(args.data):
        raise ValueError("--models e --data devem ter a mesma quantidade de caminhos.")
    results: list[dict[str, Any]] = []
    for model_path, data_path in zip(args.models, args.data, strict=True):
        model_id = experiment_name(model_path)
        model = YOLO(str(model_path))
        for confidence in args.confidences:
            metrics = model.val(
                data=str(data_path),
                split="val",
                imgsz=320,
                conf=confidence,
                device="cpu",
                plots=False,
                verbose=False,
                project=str(args.output / "runs"),
                name=f"{model_id}-conf{confidence:.2f}",
                exist_ok=True,
            )
            results.append(
                {
                    "model_experiment": model_id,
                    "data": str(data_path),
                    "confidence": confidence,
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "internal_threshold_sweep.json").write_text(
        json.dumps(
            {
                "experiment_id": args.output.name,
                "version": "V10_internal_threshold_calibration",
                "selection_rule": (
                    "Escolher threshold comum usando somente validacoes internas; "
                    "conjunto externo permanece reservado."
                ),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
