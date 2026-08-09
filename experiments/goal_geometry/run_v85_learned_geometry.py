"""Train interpretable learned geometry models with leave-one-match-out folds."""

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
    Segment,
    candidate_pool,
    frame_path,
    line_support,
    make_segments,
    morphology,
    threshold_mask,
)

FEATURES = [
    "width_height_ratio",
    "left_post_angle",
    "right_post_angle",
    "post_angle_difference",
    "height_similarity",
    "crossbar_angle",
    "relative_width",
    "relative_height",
    "relative_area",
    "perspective_skew",
    "connection_score",
    "white_support",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V8.5 learned geometry.")
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def point_features(
    row: dict[str, str], image: np.ndarray, frame_width: int, frame_height: int
) -> dict[str, float]:
    lt = (float(row["lt_x"]), float(row["lt_y"]))
    rt = (float(row["rt_x"]), float(row["rt_y"]))
    lb = (float(row["lb_x"]), float(row["lb_y"]))
    rb = (float(row["rb_x"]), float(row["rb_y"]))
    left_height = distance(lt, lb)
    right_height = distance(rt, rb)
    width = distance(lt, rt)
    mean_height = max((left_height + right_height) / 2, 1.0)
    left_segment = Segment(*lt, *lb, 0.0, left_height)
    right_segment = Segment(*rt, *rb, 0.0, right_height)
    bar_segment = Segment(*lt, *rt, 0.0, width)
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 190, 255, cv2.THRESH_BINARY)
    white_support = np.mean(
        [
            line_support(mask, left_segment),
            line_support(mask, right_segment),
            line_support(mask, bar_segment),
        ]
    )
    left_angle = float(np.degrees(np.arctan2(lb[1] - lt[1], lb[0] - lt[0])))
    right_angle = float(np.degrees(np.arctan2(rb[1] - rt[1], rb[0] - rt[0])))
    return {
        "width_height_ratio": width / mean_height,
        "left_post_angle": left_angle,
        "right_post_angle": right_angle,
        "post_angle_difference": abs(left_angle - right_angle),
        "height_similarity": min(left_height, right_height) / max(left_height, right_height, 1.0),
        "crossbar_angle": float(np.degrees(np.arctan2(rt[1] - lt[1], rt[0] - lt[0]))),
        "relative_width": width / frame_width,
        "relative_height": mean_height / frame_height,
        "relative_area": (width * mean_height) / (frame_width * frame_height),
        "perspective_skew": abs((lt[1] + rt[1]) / 2 - (lb[1] + rb[1]) / 2) / mean_height,
        "connection_score": 1.0,
        "white_support": float(white_support),
    }


def candidate_features(candidate: Any, frame_width: int, frame_height: int) -> dict[str, float]:
    left = candidate.left_post
    right = candidate.right_post
    bar = candidate.crossbar
    mean_height = max((left.length + right.length) / 2, 1.0)
    return {
        "width_height_ratio": candidate.width_height_ratio,
        "left_post_angle": left.angle,
        "right_post_angle": right.angle,
        "post_angle_difference": abs(left.angle - right.angle),
        "height_similarity": candidate.height_score,
        "crossbar_angle": bar.angle,
        "relative_width": candidate.relative_width,
        "relative_height": candidate.relative_height,
        "relative_area": (candidate.width * mean_height) / (frame_width * frame_height),
        "perspective_skew": abs(bar.y1 - bar.y2) / max(candidate.width, 1.0),
        "connection_score": candidate.connection_score,
        "white_support": candidate.white_support_score,
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    expected = [row["true"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    tp = sum(a and b for a, b in zip(expected, predicted, strict=True))
    fp = sum(not a and b for a, b in zip(expected, predicted, strict=True))
    tn = sum(not a and not b for a, b in zip(expected, predicted, strict=True))
    fn = sum(a and not b for a, b in zip(expected, predicted, strict=True))
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
    timestamps = list(csv.DictReader(args.timestamps.open(newline="", encoding="utf-8")))
    annotations = list(csv.DictReader(args.annotations.open(newline="", encoding="utf-8")))
    if len(annotations) != 35 or any(
        row["annotation_status"] != "completed" for row in annotations
    ):
        raise ValueError("V8.5 exige 35 anotacoes completas")
    args.output.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for annotation in annotations:
        image = cv2.imread(annotation["image_path"])
        if image is None:
            raise FileNotFoundError(annotation["image_path"])
        features = point_features(annotation, image, image.shape[1], image.shape[0])
        samples.append(
            {
                "sample_id": annotation["frame_id"],
                "frame_id": annotation["frame_id"],
                "match_id": annotation["match_id"],
                "label": "goal_structure",
                "source": "manual_annotation",
                **features,
            }
        )
    for row in timestamps:
        if row["label"] != "not_goal":
            continue
        image = cv2.imread(str(frame_path(args.frames_dir, row)))
        if image is None:
            raise FileNotFoundError(frame_path(args.frames_dir, row))
        scale = min(1.0, 960.0 / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = morphology(threshold_mask(image, config), config)
        _edges, segments = make_segments(mask, config)
        candidates = candidate_pool(mask, segments, config, merged=True)
        frame_id = f"{row['match_id']}_{float(row['timestamp_seconds']):010.3f}"
        for index, candidate in enumerate(candidates):
            samples.append(
                {
                    "sample_id": f"{frame_id}_c{index + 1:03d}",
                    "frame_id": frame_id,
                    "match_id": row["match_id"],
                    "label": "not_goal",
                    "source": "hough_candidate",
                    **candidate_features(candidate, image.shape[1], image.shape[0]),
                }
            )
    with (args.output / "goal_geometry_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = ["sample_id", "frame_id", "match_id", "label", "source", *FEATURES]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)
    models: dict[str, Any] = {
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
    frame_truth = {
        row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}": row["label"]
        in {"main_goal", "secondary_goal"}
        for row in timestamps
    }
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    match_ids = sorted({row["match_id"] for row in timestamps})
    start = time.perf_counter()
    for model_name, model in models.items():
        for test_match in match_ids:
            train = [row for row in samples if row["match_id"] != test_match]
            test = [row for row in samples if row["match_id"] == test_match]
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
                score = frame_scores.get(frame_id, 0.0)
                predicted = score >= 0.5
                frame_eval.append(
                    {"frame_id": frame_id, "true": truth, "predicted": predicted, "score": score}
                )
                prediction_rows.append({"model": model_name, "fold": test_match, **frame_eval[-1]})
            fold_rows.append({"model": model_name, "test_match": test_match, **metrics(frame_eval)})
    model_rows = []
    for model_name in models:
        model_eval = [row for row in fold_rows if row["model"] == model_name]
        model_rows.append(
            {
                "model": model_name,
                **{
                    key: float(np.mean([row[key] for row in model_eval]))
                    for key in ["precision", "recall", "f1"]
                },
                "folds": len(model_eval),
            }
        )
    with (args.output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    with (args.output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    best = max(model_rows, key=lambda row: float(str(row["f1"])))
    report = {
        "experiment_id": "EXP-0021",
        "version": "V8.5",
        "task": "goal_structure",
        "samples": len(samples),
        "models": model_rows,
        "best_model": best,
        "evaluation": "leave_one_match_out",
        "execution_seconds": time.perf_counter() - start,
        "limitations": [
            "O dataset de features ainda é pequeno; resultados são piloto.",
            "Negativos são candidatos Hough dos frames not_goal.",
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "sklearn": "installed",
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
    (args.output / "README.md").write_text(
        "# EXP-0021 - V8.5 Learned Geometry\n\n"
        "Classificadores tabulares avaliados com leave-one-match-out.\n",
        encoding="utf-8",
    )
    print(json.dumps({"experiment_id": "EXP-0021", "best_model": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
