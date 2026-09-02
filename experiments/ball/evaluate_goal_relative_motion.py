"""Mede o movimento da candidata em coordenadas normalizadas pela goleira detectada."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia deslocamento relativo da candidata à goleira.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--tracking-results", type=Path, required=True)
    parser.add_argument("--goal-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequences = list(csv.DictReader(file))
    with args.tracking_results.open(newline="", encoding="utf-8") as file:
        tracking = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sequences:
        grouped[row["sequence_id"]].append(row)
    model = YOLO(str(args.goal_model))
    frame_rows, summary_rows = [], []
    for sequence_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["sequence_frame"]))
        prior_goal = None
        relative = []
        for row in rows:
            image = cv2.imread(row["image_path"])
            result = model.predict(source=image, conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            goals = [list(map(float, box)) for box in result.boxes.xyxy.tolist()]
            selected_json = tracking[(sequence_id, row["sequence_frame"])].get("selected_box_json", "")
            selected = json.loads(selected_json) if selected_json else None
            goal = None
            if goals:
                reference = center(prior_goal) if prior_goal else (center(selected) if selected else center(goals[0]))
                goal = min(goals, key=lambda box: math.dist(center(box), reference))
                prior_goal = goal
            normalized = None
            if selected and goal and goal[2] > goal[0] and goal[3] > goal[1]:
                px, py = center(selected)
                normalized = ((px - goal[0]) / (goal[2] - goal[0]), (py - goal[1]) / (goal[3] - goal[1]))
                relative.append(normalized)
            frame_rows.append({"sequence_id": sequence_id, "sequence_frame": row["sequence_frame"], "event_id": row["event_id"], "match_id": row["match_id"], "goal_box_json": json.dumps(goal) if goal else "", "relative_x": f"{normalized[0]:.6f}" if normalized else "", "relative_y": f"{normalized[1]:.6f}" if normalized else ""})
        steps = [math.dist(relative[index - 1], relative[index]) for index in range(1, len(relative))]
        summary_rows.append({"sequence_id": sequence_id, "event_id": rows[0]["event_id"], "match_id": rows[0]["match_id"], "relative_frames": str(len(relative)), "mean_relative_step": f"{sum(steps) / len(steps):.6f}" if steps else "", "max_relative_step": f"{max(steps):.6f}" if steps else "", "net_relative_displacement": f"{math.dist(relative[0], relative[-1]):.6f}" if len(relative) >= 2 else ""})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("goal_relative_motion_frames.csv", frame_rows), ("goal_relative_motion_summary.csv", summary_rows)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"sequencias={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
