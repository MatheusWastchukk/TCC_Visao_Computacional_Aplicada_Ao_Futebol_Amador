"""Train V8.5 on Hough candidates labeled by manual quadrilateral overlap."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from experiments.goal_geometry.run_geometry_versions import (
    apply_exclusion_zones,
    candidate_pool,
    frame_path,
    make_segments,
    morphology,
    threshold_mask,
)
from experiments.goal_geometry.run_v85_learned_geometry import (
    FEATURES,
    candidate_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V8.5 with candidate-level labels.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument(
        "--models",
        default="logistic_regression,decision_tree,random_forest,svm",
        help="Comma-separated model names to execute.",
    )
    return parser.parse_args()


def bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def candidate_box(candidate: Any) -> tuple[float, float, float, float]:
    points: list[tuple[float, float]] = []
    for segment in (candidate.left_post, candidate.right_post, candidate.crossbar):
        points.extend([(segment.x1, segment.y1), (segment.x2, segment.y2)])
    return bbox(points)


def models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=3, random_state=42, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=42, class_weight="balanced"
        ),
        "svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
    }


def frame_candidates(image: np.ndarray, config: dict[str, Any]) -> list[Any]:
    max_dimension = config.get("processing", {}).get("max_dimension", 960.0)
    scale = min(1.0, float(max_dimension) / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = apply_exclusion_zones(morphology(threshold_mask(image, config), config), config)
    _edges, segments = make_segments(mask, config)
    return candidates_for_image(image, segments, mask, config)


def candidates_for_image(
    image: np.ndarray, segments: list[Any], mask: np.ndarray, config: dict[str, Any]
) -> list[Any]:
    return candidate_pool(mask, segments, config, merged=True)


def candidate_rows(
    image: np.ndarray,
    candidates: list[Any],
    frame_id: str,
    match_id: str,
    labeler: Any | None = None,
    iou_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(candidates):
        row: dict[str, Any] = {
            "sample_id": f"{frame_id}_c{index + 1:03d}",
            "frame_id": frame_id,
            "match_id": match_id,
            **candidate_features(candidate, image.shape[1], image.shape[0]),
        }
        if labeler is not None:
            row["candidate_iou"] = labeler(candidate)
            row["label"] = "goal_structure" if row["candidate_iou"] >= iou_threshold else "not_goal"
        rows.append(row)
    return rows


def frame_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    true = [bool(row["true"]) for row in rows]
    predicted = [bool(row["predicted"]) for row in rows]
    tp = sum(a and b for a, b in zip(true, predicted, strict=True))
    fp = sum(not a and b for a, b in zip(true, predicted, strict=True))
    tn = sum(not a and not b for a, b in zip(true, predicted, strict=True))
    fn = sum(a and not b for a, b in zip(true, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    annotations = list(csv.DictReader(args.annotations.open(newline="", encoding="utf-8")))
    timestamps = list(csv.DictReader(args.timestamps.open(newline="", encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for annotation in annotations:
        image = cv2.imread(annotation["image_path"])
        if image is None:
            raise FileNotFoundError(annotation["image_path"])
        max_dimension = config.get("processing", {}).get("max_dimension", 960.0)
        scale = min(1.0, float(max_dimension) / max(image.shape[:2]))
        points = [
            (float(annotation[key]) * scale, float(annotation[key_y]) * scale)
            for key, key_y in [
                ("lt_x", "lt_y"),
                ("rt_x", "rt_y"),
                ("lb_x", "lb_y"),
                ("rb_x", "rb_y"),
            ]
        ]
        ground_truth = bbox(points)
        resized = image
        if scale < 1.0:
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = apply_exclusion_zones(morphology(threshold_mask(resized, config), config), config)
        _edges, segments = make_segments(mask, config)
        candidates = candidate_pool(mask, segments, config, merged=True)
        frame_id = annotation["frame_id"]
        rows = candidate_rows(
            resized,
            candidates,
            frame_id,
            annotation["match_id"],
            lambda candidate, gt=ground_truth: bbox_iou(gt, candidate_box(candidate)),
            args.iou_threshold,
        )
        training_rows.extend(rows)
        positive_count = sum(row["label"] == "goal_structure" for row in rows)
        coverage.append(
            {
                "frame_id": frame_id,
                "match_id": annotation["match_id"],
                "candidates": len(rows),
                "positive_candidates": positive_count,
                "max_iou": max((row["candidate_iou"] for row in rows), default=0.0),
            }
        )
    external_rows: list[dict[str, Any]] = []
    frame_truth: dict[str, bool] = {}
    for timestamp in timestamps:
        image = cv2.imread(str(frame_path(args.frames_dir, timestamp)))
        if image is None:
            raise FileNotFoundError(frame_path(args.frames_dir, timestamp))
        frame_id = f"{timestamp['match_id']}_{float(timestamp['timestamp_seconds']):010.3f}"
        candidates = frame_candidates(image, config)
        external_rows.extend(candidate_rows(image, candidates, frame_id, timestamp["match_id"]))
        frame_truth[frame_id] = timestamp["label"] in {"main_goal", "secondary_goal"}
    fields = ["sample_id", "frame_id", "match_id", "label", "candidate_iou", *FEATURES]
    with (args.output / "candidate_training.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(training_rows)
    with (args.output / "annotation_coverage.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    match_ids = sorted({row["match_id"] for row in timestamps})
    start = time.perf_counter()
    selected_models = models()
    requested_models = [name.strip() for name in args.models.split(",") if name.strip()]
    selected_models = {name: selected_models[name] for name in requested_models}
    for model_name, model in selected_models.items():
        for test_match in match_ids:
            train = [row for row in training_rows if row["match_id"] != test_match]
            test = [row for row in external_rows if row["match_id"] == test_match]
            x_train = np.array([[row[feature] for feature in FEATURES] for row in train])
            y_train = np.array([row["label"] for row in train])
            x_test = np.array([[row[feature] for feature in FEATURES] for row in test])
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[
                :, list(model.classes_).index("goal_structure")
            ]
            frame_scores: dict[str, float] = {}
            for sample, probability in zip(test, probabilities, strict=True):
                frame_scores[sample["frame_id"]] = max(
                    frame_scores.get(sample["frame_id"], 0.0), float(probability)
                )
            frame_eval = []
            for frame_id, truth in frame_truth.items():
                if not frame_id.startswith(f"{test_match}_"):
                    continue
                row = {
                    "frame_id": frame_id,
                    "true": truth,
                    "predicted": frame_scores.get(frame_id, 0.0) >= 0.5,
                    "score": frame_scores.get(frame_id, 0.0),
                }
                frame_eval.append(row)
                prediction_rows.append({"model": model_name, "fold": test_match, **row})
            fold_rows.append(
                {"model": model_name, "test_match": test_match, **frame_metrics(frame_eval)}
            )
        model_folds = [row for row in fold_rows if row["model"] == model_name]
        model_rows.append(
            {
                "model": model_name,
                **{
                    key: float(np.mean([row[key] for row in model_folds]))
                    for key in ["precision", "recall", "f1"]
                },
                "folds": len(model_folds),
            }
        )
    for _name, rows, path in [
        ("fold_metrics", fold_rows, args.output / "fold_metrics.csv"),
        ("model_comparison", model_rows, args.output / "model_comparison.csv"),
        ("predictions", prediction_rows, args.output / "predictions.csv"),
    ]:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    best = max(model_rows, key=lambda row: float(str(row["f1"])))
    report = {
        "experiment_id": args.output.name,
        "version": "V8.5_candidate_level",
        "training_frames": len(annotations),
        "training_candidates": len(training_rows),
        "external_frames": len(frame_truth),
        "iou_threshold": args.iou_threshold,
        "models": model_rows,
        "best_model": best,
        "evaluation": "leave_one_match_out_on_external_frames",
        "execution_seconds": time.perf_counter() - start,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip(),
                "date": str(date.today()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": args.output.name,
                "best_model": best,
                "training_candidates": len(training_rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
