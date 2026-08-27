"""Audita previsões do baseline V10 no conjunto validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def ground_truth(row: dict[str, str]) -> list[tuple[float, float, float, float]]:
    image_width = float(row["width"])
    image_height = float(row["height"])
    boxes = []
    for line in Path(row["label_path"]).read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        _, center_x, center_y, box_width, box_height = map(float, values[:5])
        width = box_width * image_width
        height = box_height * image_height
        center_x *= image_width
        center_y *= image_height
        boxes.append(
            (
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            )
        )
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita o baseline V10 na validation.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--experiment-id", default="EXP-0089")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "val"]
    with args.split_manifest.open(newline="", encoding="utf-8") as file:
        capture_by_match = {
            row["match_id"]: row["capture_domain"] for row in csv.DictReader(file)
        }
    model = YOLO(str(args.model))
    output_rows: list[dict[str, object]] = []
    for row, result in zip(
        rows,
        model.predict(
            [row["image_path"] for row in rows],
            imgsz=640,
            conf=args.confidence,
            device="cpu",
            verbose=False,
        ),
        strict=True,
    ):
        truths = ground_truth(row)
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        matches = [iou(tuple(box), truth) for box in boxes for truth in truths]
        best_iou = max(matches, default=0.0)
        best_confidence = max(confidences, default=0.0)
        matched_truths = sum(
            any(iou(tuple(box), truth) >= args.iou_threshold for box in boxes)
            for truth in truths
        )
        true_positive = bool(truths) and matched_truths > 0
        false_negative = bool(truths) and matched_truths == 0
        partial_miss = bool(truths) and 0 < matched_truths < len(truths)
        false_positive = not truths and bool(boxes)
        if true_positive:
            error_reason = "none"
        elif false_negative:
            error_reason = "missed_goal_structure"
        elif partial_miss:
            error_reason = "partial_multitarget_detection"
        elif false_positive:
            error_reason = "background_confusion"
        else:
            error_reason = "none"
        output_rows.append(
            {
                "frame_id": row["frame_id"],
                "match_id": row["match_id"],
                "capture_system": capture_by_match.get(row["match_id"], "unknown"),
                "ground_truth": "goal_structure" if truths else "not_goal",
                "ground_truth_count": len(truths),
                "matched_ground_truth_count": matched_truths,
                "missed_ground_truth_count": max(0, len(truths) - matched_truths),
                "prediction_count": len(boxes),
                "best_confidence": round(best_confidence, 6),
                "best_iou": round(best_iou, 6),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "partial_miss": partial_miss,
                "error_reason": error_reason,
                "manual_review": "pending",
                "image_path": row["image_path"],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "experiment_id": args.experiment_id,
        "validation_images": len(output_rows),
        "confidence": args.confidence,
        "iou_threshold": args.iou_threshold,
        "true_positive_frames": sum(bool(row["true_positive"]) for row in output_rows),
        "false_positive_frames": sum(bool(row["false_positive"]) for row in output_rows),
        "false_negative_frames": sum(bool(row["false_negative"]) for row in output_rows),
        "partial_multitarget_frames": sum(bool(row["partial_miss"]) for row in output_rows),
        "ground_truth_structures": sum(int(row["ground_truth_count"]) for row in output_rows),
        "matched_ground_truth_structures": sum(int(row["matched_ground_truth_count"]) for row in output_rows),
        "missed_ground_truth_structures": sum(int(row["missed_ground_truth_count"]) for row in output_rows),
        "manual_review": "pending",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
