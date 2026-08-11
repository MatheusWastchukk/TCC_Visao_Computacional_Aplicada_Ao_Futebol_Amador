"""Calibrate V8.5 decision thresholds on development data, then score external predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from experiments.goal_geometry.run_geometry_versions import frame_path
from experiments.goal_geometry.run_v85_candidate_level import (
    candidate_rows,
    frame_candidates,
    frame_metrics,
    models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate V8.5 thresholds.")
    parser.add_argument("--training-candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration-timestamps", type=Path, required=True)
    parser.add_argument("--calibration-frames-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="logistic_regression,decision_tree")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def scores_by_frame(
    model: Any, rows: list[dict[str, Any]], features: list[str]
) -> dict[str, float]:
    x = np.array([[float(row[feature]) for feature in features] for row in rows])
    probabilities = model.predict_proba(x)[:, list(model.classes_).index("goal_structure")]
    scores: dict[str, float] = {}
    for row, probability in zip(rows, probabilities, strict=True):
        scores[row["frame_id"]] = max(scores.get(row["frame_id"], 0.0), float(probability))
    return scores


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = read_rows(args.training_candidates)
    annotations = read_rows(args.annotations)
    calibration_timestamps = [
        row for row in read_rows(args.calibration_timestamps) if row["label"] == "not_goal"
    ]
    features = [
        key
        for key in training[0]
        if key not in {"sample_id", "frame_id", "match_id", "label", "candidate_iou"}
    ]
    calibration_rows: list[dict[str, Any]] = []
    calibration_truth: dict[str, bool] = {}
    for annotation in annotations:
        rows = [row for row in training if row["frame_id"] == annotation["frame_id"]]
        calibration_rows.extend(rows)
        calibration_truth[annotation["frame_id"]] = True
    for timestamp in calibration_timestamps:
        image = cv2.imread(str(frame_path(args.calibration_frames_dir, timestamp)))
        if image is None:
            raise FileNotFoundError(frame_path(args.calibration_frames_dir, timestamp))
        frame_id = f"{timestamp['match_id']}_{float(timestamp['timestamp_seconds']):010.3f}"
        rows = candidate_rows(
            image,
            frame_candidates(image, config),
            frame_id,
            timestamp["match_id"],
        )
        calibration_rows.extend(rows)
        calibration_truth[frame_id] = False
    match_ids = sorted({row["match_id"] for row in calibration_rows})
    threshold_rows: list[dict[str, Any]] = []
    selected_models = models()
    for model_name in [name.strip() for name in args.models.split(",") if name.strip()]:
        for test_match in match_ids:
            train = [row for row in training if row["match_id"] != test_match]
            test = [row for row in calibration_rows if row["match_id"] == test_match]
            estimator = selected_models[model_name]
            estimator.fit(
                np.array([[float(row[feature]) for feature in features] for row in train]),
                np.array([row["label"] for row in train]),
            )
            scores = scores_by_frame(estimator, test, features)
            for threshold in np.arange(0.05, 1.0, 0.05):
                frame_eval = [
                    {
                        "frame_id": frame_id,
                        "true": truth,
                        "predicted": scores.get(frame_id, 0.0) >= threshold,
                    }
                    for frame_id, truth in calibration_truth.items()
                    if frame_id.startswith(f"{test_match}_")
                ]
                threshold_rows.append(
                    {
                        "model": model_name,
                        "fold": test_match,
                        "threshold": round(float(threshold), 2),
                        **frame_metrics(frame_eval),
                    }
                )
    aggregate: list[dict[str, Any]] = []
    for model_name in [name.strip() for name in args.models.split(",") if name.strip()]:
        for threshold in sorted({row["threshold"] for row in threshold_rows}):
            rows = [
                row
                for row in threshold_rows
                if row["model"] == model_name and row["threshold"] == threshold
            ]
            aggregate.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    **{
                        key: float(np.mean([row[key] for row in rows]))
                        for key in ["precision", "recall", "f1"]
                    },
                }
            )
    selected: dict[str, dict[str, Any]] = {}
    for model_name in [name.strip() for name in args.models.split(",") if name.strip()]:
        options = [row for row in aggregate if row["model"] == model_name]
        selected[model_name] = max(
            options,
            key=lambda row: (row["f1"], row["precision"], row["threshold"]),
        )
    external = read_rows(args.external_predictions)
    external_rows: list[dict[str, Any]] = []
    for model_name, selection in selected.items():
        rows = [row for row in external if row["model"] == model_name]
        evaluated = [
            {
                "model": model_name,
                "threshold": selection["threshold"],
                "frame_id": row["frame_id"],
                "true": row["true"].lower() == "true",
                "predicted": float(row["score"]) >= selection["threshold"],
                "score": row["score"],
            }
            for row in rows
        ]
        external_rows.append(
            {
                "model": model_name,
                "threshold": selection["threshold"],
                **frame_metrics(evaluated),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("threshold_sweep.csv", threshold_rows),
        ("threshold_aggregate.csv", aggregate),
        ("external_metrics_calibrated.csv", external_rows),
    ]:
        with (args.output / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print({"selected": selected, "external": external_rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
