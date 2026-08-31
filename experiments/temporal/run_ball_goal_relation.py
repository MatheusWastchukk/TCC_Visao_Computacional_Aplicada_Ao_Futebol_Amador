"""Relaciona tracks da bola às caixas de estrutura de gol por frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ball_center(row: dict[str, str]) -> tuple[float, float] | None:
    values = [row.get(f"track_{axis}") for axis in ("x1", "y1", "x2", "y2")]
    if any(value in {None, ""} for value in values):
        return None
    return ((float(values[0]) + float(values[2])) / 2, (float(values[1]) + float(values[3])) / 2)


def inside(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Relaciona a bola rastreada às estruturas de gol.")
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--goal-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    frames = [row for row in read_csv(args.tracking / "frame_tracking.csv") if row["track_active"] == "True" and ball_center(row)]
    model = YOLO(str(args.goal_model))
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    captures: dict[str, cv2.VideoCapture] = {}
    started = time.perf_counter()
    try:
        for index, row in enumerate(frames, start=1):
            video_name = row["video_filename"]
            capture = captures.get(video_name)
            if capture is None:
                capture = cv2.VideoCapture(str(args.videos_dir / video_name))
                if not capture.isOpened():
                    raise FileNotFoundError(args.videos_dir / video_name)
                captures[video_name] = capture
            capture.set(cv2.CAP_PROP_POS_MSEC, float(row["timestamp_seconds"]) * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            result = model.predict(source=image, imgsz=args.imgsz, conf=args.confidence, device=args.device, verbose=False)[0]
            boxes = [tuple(float(value) for value in box) for box in result.boxes.xyxy.tolist()]
            confidences = [float(value) for value in result.boxes.conf.tolist()]
            center = ball_center(row)
            nearest_index = -1
            nearest_distance = 0.0
            if boxes and center:
                distances = [math.hypot(center[0] - (box[0] + box[2]) / 2, center[1] - (box[1] + box[3]) / 2) for box in boxes]
                nearest_index = min(range(len(boxes)), key=distances.__getitem__)
                nearest_distance = distances[nearest_index]
            nearest = boxes[nearest_index] if nearest_index >= 0 else None
            output_rows.append({
                "block_id": row["block_id"], "match_id": row["match_id"], "video_filename": video_name,
                "capture_domain": row["capture_domain"], "timestamp_seconds": row["timestamp_seconds"],
                "timestamp_hhmmss": row["timestamp_hhmmss"], "ball_x1": row["track_x1"], "ball_y1": row["track_y1"],
                "ball_x2": row["track_x2"], "ball_y2": row["track_y2"], "goal_prediction_count": len(boxes),
                "nearest_goal_confidence": f"{confidences[nearest_index]:.6f}" if nearest else "",
                "nearest_goal_x1": f"{nearest[0]:.2f}" if nearest else "", "nearest_goal_y1": f"{nearest[1]:.2f}" if nearest else "",
                "nearest_goal_x2": f"{nearest[2]:.2f}" if nearest else "", "nearest_goal_y2": f"{nearest[3]:.2f}" if nearest else "",
                "ball_goal_center_distance": f"{nearest_distance:.3f}" if nearest else "",
                "ball_inside_nearest_goal": inside(center, nearest) if nearest and center else False,
            })
            if index % 500 == 0 or index == len(frames):
                print(f"Processados: {index}/{len(frames)}")
    finally:
        for capture in captures.values():
            capture.release()
    with (args.output / "frame_ball_goal_relation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "experiment_id": args.output.name,
        "tracking": str(args.tracking),
        "goal_model": str(args.goal_model),
        "tracked_ball_frames": len(output_rows),
        "frames_with_goal_structure": sum(row["goal_prediction_count"] > 0 for row in output_rows),
        "ball_inside_goal_frames": sum(bool(row["ball_inside_nearest_goal"]) for row in output_rows),
        "execution_seconds": round(time.perf_counter() - started, 3),
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
