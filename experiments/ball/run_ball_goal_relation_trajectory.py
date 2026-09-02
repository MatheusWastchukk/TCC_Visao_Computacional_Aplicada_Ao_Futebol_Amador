"""Relaciona a trajetória selecionada da bola às estruturas de gol detectadas por frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def inside(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Relaciona trajetórias selecionadas da bola a goleiras detectadas.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--tracking-results", type=Path, required=True)
    parser.add_argument("--quality-review", type=Path, required=True)
    parser.add_argument("--goal-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequences = list(csv.DictReader(file))
    with args.tracking_results.open(newline="", encoding="utf-8") as file:
        tracking = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    with args.quality_review.open(newline="", encoding="utf-8") as file:
        usable = {row["sequence_id"] for row in csv.DictReader(file) if row["manual_quality"] == "UTIL"}
    rows = [row for row in sequences if row["sequence_id"] in usable]
    if not rows:
        raise ValueError("Não há trajetórias classificadas como UTIL.")
    model = YOLO(str(args.goal_model))
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        tracker = tracking[(row["sequence_id"], row["sequence_frame"])]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        selected = tuple(float(value) for value in json.loads(tracker["selected_box_json"])) if tracker.get("selected_box_json") else None
        result = model.predict(source=image, conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = [tuple(float(value) for value in box) for box in result.boxes.xyxy.tolist()]
        confidences = [float(value) for value in result.boxes.conf.tolist()]
        nearest_index = -1
        distance = None
        if selected and boxes:
            selected_center = center(selected)
            distances = [math.hypot(selected_center[0] - center(box)[0], selected_center[1] - center(box)[1]) for box in boxes]
            nearest_index = min(range(len(distances)), key=distances.__getitem__)
            distance = distances[nearest_index]
        nearest = boxes[nearest_index] if nearest_index >= 0 else None
        output_rows.append({
            "sequence_id": row["sequence_id"], "sequence_frame": row["sequence_frame"], "sample_id": row["sample_id"],
            "match_id": row["match_id"], "image_path": row["image_path"], "target_box_json": row["ball_of_play_box_json"],
            "selected_ball_box_json": tracker.get("selected_box_json", ""), "identity_correct": tracker["identity_correct"],
            "goal_prediction_count": str(len(boxes)), "nearest_goal_box_json": json.dumps(nearest) if nearest else "",
            "nearest_goal_confidence": f"{confidences[nearest_index]:.6f}" if nearest else "",
            "ball_goal_center_distance": f"{distance:.3f}" if distance is not None else "",
            "ball_inside_nearest_goal": str(inside(center(selected), nearest)) if selected and nearest else "",
        })
        print(f"Processado: {index}/{len(rows)}")
    with (args.output / "frame_ball_goal_relation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    metrics = {
        "experiment_id": args.output.name,
        "usable_sequences": len(usable), "frames": len(output_rows),
        "frames_with_goal_prediction": sum(row["goal_prediction_count"] != "0" for row in output_rows),
        "ball_inside_goal_frames": sum(row["ball_inside_nearest_goal"] == "True" for row in output_rows),
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
