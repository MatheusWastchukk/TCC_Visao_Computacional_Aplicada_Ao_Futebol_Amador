"""Calibra o ranqueamento de propostas da V8.6 com as anotações de desenvolvimento."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.goal_geometry.run_v85_candidate_level import bbox, bbox_iou, candidate_box
from experiments.goal_geometry.run_v85_learned_geometry import FEATURES
from experiments.goal_geometry.run_v86_proposal_generator import (
    frame_path,
    limits_from_annotations,
    proposal_rows,
    select_proposals,
)

RANK_FEATURES = [
    *FEATURES,
    "topology_score",
    "box_brightness",
    "box_white_fraction",
    "box_green_fraction",
    "box_blue_fraction",
    "box_edge_density",
    "upper_blue_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o ranqueamento aprendido de propostas da V8.6.")
    parser.add_argument("--development-annotations", type=Path, required=True)
    parser.add_argument("--external-annotations", type=Path, required=True)
    parser.add_argument("--external-timestamps", type=Path, required=True)
    parser.add_argument("--external-frames-dir", type=Path, required=True)
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


def grouped(rows_in: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows_in:
        groups.setdefault(row["frame_id"], []).append(row)
    return list(groups.values())


def frame_data(
    annotation_group: list[dict[str, str]],
    image_path: Path,
    config: dict[str, Any],
    limits: dict[str, float],
) -> tuple[list[dict[str, Any]], list[tuple[float, float, float, float]]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    scale = min(1.0, 1920.0 / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    truths = [truth_box(row, scale) for row in annotation_group]
    proposals = proposal_rows(image, config, limits)
    return proposals, truths


def model_set() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"
        ),
    }


def feature_matrix(proposals: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[float(item["features"][feature]) for feature in RANK_FEATURES] for item in proposals]
    )


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


def evaluate(
    model: Any,
    external: list[
        tuple[str, list[dict[str, Any]], list[tuple[float, float, float, float]]]
    ],
) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    for frame_id, proposals, truths in external:
        probabilities = model.predict_proba(feature_matrix(proposals))[
            :, list(model.classes_).index("goal")
        ]
        ranked = []
        for probability, item in sorted(
            zip(probabilities, proposals, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        ):
            ranked_item = dict(item)
            ranked_item["proposal_score"] = float(probability)
            ranked.append(ranked_item)
        selected = select_proposals(
            ranked,
            {"proposal_selection": {"max_candidates": 50, "nms_iou_threshold": 0.70}},
        )
        values = {f"top{k}_iou": iou_at(selected, truths, k) for k in [1, 5, 10]}
        rows_out.append(
            {
                "frame_id": frame_id,
                "num_candidates": len(selected),
                "top1_iou": values["top1_iou"],
                "top5_iou": values["top5_iou"],
                "top10_iou": values["top10_iou"],
                **{f"top{k}_iou_ge_0_30": values[f"top{k}_iou"] >= 0.30 for k in [1, 5, 10]},
                **{f"top{k}_iou_ge_0_50": values[f"top{k}_iou"] >= 0.50 for k in [1, 5, 10]},
            }
        )
    return {
        "frames": len(rows_out),
        "top1_recall_iou_030": float(np.mean([row["top1_iou_ge_0_30"] for row in rows_out])),
        "top5_recall_iou_030": float(np.mean([row["top5_iou_ge_0_30"] for row in rows_out])),
        "top10_recall_iou_030": float(np.mean([row["top10_iou_ge_0_30"] for row in rows_out])),
        "top1_recall_iou_050": float(np.mean([row["top1_iou_ge_0_50"] for row in rows_out])),
        "top5_recall_iou_050": float(np.mean([row["top5_iou_ge_0_50"] for row in rows_out])),
        "top10_recall_iou_050": float(np.mean([row["top10_iou_ge_0_50"] for row in rows_out])),
        "mean_candidates": float(np.mean([row["num_candidates"] for row in rows_out])),
        "mean_top1_iou": float(np.mean([row["top1_iou"] for row in rows_out])),
        "mean_top5_iou": float(np.mean([row["top5_iou"] for row in rows_out])),
        "rows": rows_out,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    development = rows(args.development_annotations)
    external_annotations = rows(args.external_annotations)
    timestamps = {
        row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}": row
        for row in rows(args.external_timestamps)
    }
    limits = limits_from_annotations(development)
    training: list[dict[str, Any]] = []
    for group in grouped(development):
        proposals, truths = frame_data(group, Path(group[0]["image_path"]), config, limits)
        for proposal in proposals:
            proposal["label"] = "goal" if iou_at([proposal], truths, 1) >= 0.30 else "not_goal"
            training.append(proposal)
    external: list[tuple[str, list[dict[str, Any]], list[tuple[float, float, float, float]]]] = []
    for group in grouped(external_annotations):
        annotation = group[0]
        timestamp = timestamps[annotation["frame_id"]]
        proposals, truths = frame_data(
            group, frame_path(args.external_frames_dir, timestamp), config, limits
        )
        external.append((annotation["frame_id"], proposals, truths))
    x_train = feature_matrix(training)
    y_train = np.array([row["label"] for row in training])
    args.output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {
        "experiment_id": args.output.name,
        "version": "V8.6_learned_ranker",
        "training_candidates": len(training),
    }
    for name, model in model_set().items():
        model.fit(x_train, y_train)
        reports[name] = evaluate(model, external)
        rows_out = reports[name].pop("rows")
        with (args.output / f"{name}_external.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
    (args.output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
