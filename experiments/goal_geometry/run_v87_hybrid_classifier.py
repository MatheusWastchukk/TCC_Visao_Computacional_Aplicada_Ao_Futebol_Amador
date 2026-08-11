"""Treina e avalia um classificador visual leve nos recortes de propostas da V8.6."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.goal_geometry.prepare_v87_crops import crop_with_padding
from experiments.goal_geometry.run_v85_candidate_level import bbox, bbox_iou, candidate_box
from experiments.goal_geometry.run_v86_proposal_generator import (
    frame_path,
    limits_from_annotations,
    proposal_rows,
    select_proposals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o classificador visual híbrido da V8.7.")
    parser.add_argument("--development-annotations", type=Path, required=True)
    parser.add_argument("--external-annotations", type=Path, required=True)
    parser.add_argument("--external-timestamps", type=Path, required=True)
    parser.add_argument("--external-frames-dir", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def truth_box(row: dict[str, str], scale: float = 1.0) -> tuple[float, float, float, float]:
    return bbox(
        [
            (float(row[x]) * scale, float(row[y]) * scale)
            for x, y in [("lt_x", "lt_y"), ("rt_x", "rt_y"), ("lb_x", "lb_y"), ("rb_x", "rb_y")]
        ]
    )


def grouped(annotation_rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in annotation_rows:
        groups.setdefault(row["frame_id"], []).append(row)
    return list(groups.values())


def visual_features(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray_small /= 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    gradient = cv2.resize(gradient, (16, 16), interpolation=cv2.INTER_AREA)
    gradient_normalized = gradient / max(float(gradient.max()), 1.0)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histograms = [
        cv2.calcHist([hsv], [channel], None, [16], [0, 256]).reshape(-1)
        for channel in range(3)
    ]
    histogram = np.concatenate(histograms).astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    return np.concatenate([gray_small.reshape(-1), gradient_normalized.reshape(-1), histogram])


def feature_matrix(images: list[np.ndarray]) -> np.ndarray:
    return np.vstack([visual_features(image) for image in images])


def iou_at(
    proposals: list[dict[str, Any]],
    truths: list[tuple[float, float, float, float]],
    limit: int,
) -> float:
    return max(
        (
            bbox_iou(truth, candidate_box(item["candidate"]))
            for item in proposals[:limit]
            for truth in truths
        ),
        default=0.0,
    )


def models() -> dict[str, Any]:
    return {
        "logistic_visual_color": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "svm_visual_color": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
    }


def evaluate(
    model: Any,
    external: list[
        tuple[
            str,
            list[dict[str, Any]],
            list[tuple[float, float, float, float]],
            list[np.ndarray],
        ]
    ],
) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    for frame_id, proposals, truths, crops in external:
        probabilities = model.predict_proba(feature_matrix(crops))[
            :, list(model.classes_).index("goal")
        ]
        ranked: list[dict[str, Any]] = []
        for probability, proposal in sorted(
            zip(probabilities, proposals, strict=True), key=lambda pair: pair[0], reverse=True
        ):
            item = dict(proposal)
            item["proposal_score"] = float(probability)
            ranked.append(item)
        selected = select_proposals(
            ranked,
            {"proposal_selection": {"max_candidates": 50, "nms_iou_threshold": 0.70}},
        )
        values = {f"top{k}_iou": iou_at(selected, truths, k) for k in [1, 5, 10]}
        result_rows.append(
            {
                "frame_id": frame_id,
                "num_candidates": len(selected),
                **values,
                **{f"top{k}_iou_ge_0_30": values[f"top{k}_iou"] >= 0.30 for k in [1, 5, 10]},
                **{f"top{k}_iou_ge_0_50": values[f"top{k}_iou"] >= 0.50 for k in [1, 5, 10]},
            }
        )
    return {
        "frames": len(result_rows),
        "top1_recall_iou_030": float(np.mean([row["top1_iou_ge_0_30"] for row in result_rows])),
        "top5_recall_iou_030": float(np.mean([row["top5_iou_ge_0_30"] for row in result_rows])),
        "top10_recall_iou_030": float(np.mean([row["top10_iou_ge_0_30"] for row in result_rows])),
        "top1_recall_iou_050": float(np.mean([row["top1_iou_ge_0_50"] for row in result_rows])),
        "top5_recall_iou_050": float(np.mean([row["top5_iou_ge_0_50"] for row in result_rows])),
        "top10_recall_iou_050": float(np.mean([row["top10_iou_ge_0_50"] for row in result_rows])),
        "mean_candidates": float(np.mean([row["num_candidates"] for row in result_rows])),
        "mean_top1_iou": float(np.mean([row["top1_iou"] for row in result_rows])),
        "mean_top5_iou": float(np.mean([row["top5_iou"] for row in result_rows])),
        "rows": result_rows,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    development = rows(args.development_annotations)
    external_annotations = rows(args.external_annotations)
    limits = limits_from_annotations(development)
    manifest = rows(args.crops / "manifest.csv")
    training_images = [cv2.imread(row["crop_path"]) for row in manifest]
    if any(image is None for image in training_images):
        raise FileNotFoundError("Há crops ausentes no manifesto da V8.7")
    x_train = feature_matrix([image for image in training_images if image is not None])
    y_train = np.array(
        ["goal" if row["label"] == "goal_structure" else "not_goal" for row in manifest]
    )
    timestamps = {
        row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}": row
        for row in rows(args.external_timestamps)
    }
    external: list[
        tuple[
            str,
            list[dict[str, Any]],
            list[tuple[float, float, float, float]],
            list[np.ndarray],
        ]
    ] = []
    for group in grouped(external_annotations):
        annotation = group[0]
        timestamp = timestamps[annotation["frame_id"]]
        image_path = frame_path(args.external_frames_dir, timestamp)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        truths = [truth_box(row) for row in group]
        proposals = proposal_rows(image, config, limits)
        crops = [
            crop_with_padding(image, candidate_box(item["candidate"]), 224)
            for item in proposals
        ]
        external.append((annotation["frame_id"], proposals, truths, crops))
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "experiment_id": args.output.name,
        "version": "V8.7_hybrid_visual_classifier",
        "training_crops": len(manifest),
        "positive_training_crops": sum(row["label"] == "goal_structure" for row in manifest),
        "negative_training_crops": sum(row["label"] == "not_goal" for row in manifest),
        "positive_rule": "candidate IoU >= 0.30",
    }
    for name, model in models().items():
        model.fit(x_train, y_train)
        result = evaluate(model, external)
        result_rows = result.pop("rows")
        report[name] = result
        with (args.output / f"{name}_external.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
            writer.writeheader()
            writer.writerows(result_rows)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
