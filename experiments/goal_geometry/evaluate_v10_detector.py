"""Avalia um detector V10 em frames positivos e negativos externos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from ultralytics import YOLO

from experiments.goal_geometry.run_v85_candidate_level import bbox, bbox_iou
from experiments.goal_geometry.run_v86_proposal_generator import frame_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V10 on external frames.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--negative-timestamps",
        type=Path,
        help="CSV com match_id e timestamp_seconds dos frames not_goal.",
    )
    parser.add_argument("--overlays", action="store_true")
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def grouped(annotation_rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in annotation_rows:
        grouped_rows.setdefault(row["frame_id"], []).append(row)
    return list(grouped_rows.values())


def truth_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return bbox(
        [
            (float(row[x]), float(row[y]))
            for x, y in [("lt_x", "lt_y"), ("rt_x", "rt_y"), ("lb_x", "lb_y"), ("rb_x", "rb_y")]
        ]
    )


def match_metrics(result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for match_id in sorted({str(row["match_id"]) for row in result_rows}):
        current = [row for row in result_rows if row["match_id"] == match_id]
        positives = [row for row in current if row["has_goal"]]
        negatives = [row for row in current if not row["has_goal"]]
        metrics[match_id] = {
            "frames": len(current),
            "positive_frames": len(positives),
            "negative_frames": len(negatives),
            "positive_recall_iou_030": float(
                np.mean([row["iou_ge_0_30"] for row in positives])
            )
            if positives
            else None,
            "positive_recall_iou_050": float(
                np.mean([row["iou_ge_0_50"] for row in positives])
            )
            if positives
            else None,
            "mean_best_iou": float(np.mean([row["best_iou"] for row in positives]))
            if positives
            else None,
            "negative_true_rate": float(
                np.mean([row["predictions"] == 0 for row in negatives])
            )
            if negatives
            else None,
            "negative_false_positive_rate": float(
                np.mean([row["predictions"] > 0 for row in negatives])
            )
            if negatives
            else None,
            "mean_predictions": float(np.mean([row["predictions"] for row in current])),
        }
    return metrics


def main() -> int:
    args = parse_args()
    model = YOLO(str(args.model))
    annotations = rows(args.annotations)
    timestamps = {
        row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}": row
        for row in rows(args.timestamps)
    }
    negative_rows = []
    if args.negative_timestamps:
        negative_rows = [
            row
            for row in rows(args.negative_timestamps)
            if row.get("label") == "not_goal"
        ]
    annotation_frame_ids = {group[0]["frame_id"] for group in grouped(annotations)}
    negative_rows = [
        row
        for row in negative_rows
        if row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}"
        not in annotation_frame_ids
    ]
    result_rows: list[dict[str, Any]] = []
    evaluation_groups: list[tuple[str, list[dict[str, str]], bool]] = [
        (group[0]["frame_id"], group, True) for group in grouped(annotations)
    ]
    evaluation_groups.extend(
        (
            row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}",
            [row],
            False,
        )
        for row in negative_rows
    )
    for frame_id, group, has_goal in evaluation_groups:
        annotation = group[0]
        timestamp_row = timestamps[frame_id]
        image_path = frame_path(args.frames_dir, timestamp_row)
        truths = [truth_box(row) for row in group] if has_goal else []
        predictions = cast(Any, model.predict(str(image_path), conf=args.conf, verbose=False))
        prediction = cast(Any, predictions[0])
        predicted_boxes: list[tuple[float, float, float, float]] = []
        predicted_confidences: list[float] = []
        for box in prediction.boxes.xyxy.cpu().numpy():
            values = [float(value) for value in box]
            predicted_boxes.append((values[0], values[1], values[2], values[3]))
        predicted_confidences = [
            float(value) for value in prediction.boxes.conf.cpu().numpy()
        ]
        best_iou = max(
            (bbox_iou(truth, box) for truth in truths for box in predicted_boxes),
            default=0.0,
        )
        result_rows.append(
            {
                "frame_id": frame_id,
                "match_id": annotation["match_id"],
                "has_goal": has_goal,
                "predictions": len(predicted_boxes),
                "best_iou": best_iou,
                "iou_ge_0_30": best_iou >= 0.30,
                "iou_ge_0_50": best_iou >= 0.50,
                "false_positive": not has_goal and bool(predicted_boxes),
            }
        )
        if args.overlays:
            overlay = cv2.imread(str(image_path))
            if overlay is None:
                raise FileNotFoundError(image_path)
            for truth in truths:
                cv2.rectangle(
                    overlay,
                    (int(truth[0]), int(truth[1])),
                    (int(truth[2]), int(truth[3])),
                    (0, 0, 255),
                    4,
                )
            for box, confidence in zip(predicted_boxes, predicted_confidences, strict=True):
                cv2.rectangle(
                    overlay,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    4,
                )
                cv2.putText(
                    overlay,
                    f"goal {confidence:.2f}",
                    (int(box[0]), max(25, int(box[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                overlay,
                f"{frame_id} goal={has_goal} IoU={best_iou:.3f} predictions={len(predicted_boxes)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            overlay_dir = args.output / "overlays" / annotation["match_id"]
            overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_dir / f"{frame_id}.jpg"), overlay)
    positives = [row for row in result_rows if row["has_goal"]]
    negatives = [row for row in result_rows if not row["has_goal"]]
    metrics = {
        "experiment_id": args.output.name,
        "version": "V10_external_evaluation_with_negative_frames",
        "confidence": args.conf,
        "frames": len(result_rows),
        "positive_frames": len(positives),
        "negative_frames": len(negatives),
        "recall_iou_030": float(np.mean([row["iou_ge_0_30"] for row in positives])),
        "recall_iou_050": float(np.mean([row["iou_ge_0_50"] for row in positives])),
        "mean_best_iou": float(np.mean([row["best_iou"] for row in positives])),
        "negative_true_rate": float(np.mean([row["predictions"] == 0 for row in negatives]))
        if negatives
        else None,
        "negative_false_positive_rate": float(
            np.mean([row["predictions"] > 0 for row in negatives])
        )
        if negatives
        else None,
        "mean_predictions": float(np.mean([row["predictions"] for row in result_rows])),
        "by_match": match_metrics(result_rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
