"""Executa um piloto de tracking simples para a bola em blocos contínuos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def classify(label: str, confirmed: bool) -> str:
    if label in {"inconclusive", "not_evaluable"}:
        return "inconclusive"
    if label == "goal_structure":
        return "true_positive" if confirmed else "false_negative"
    return "false_positive" if confirmed else "true_negative"


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_inside_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    """Indica se o centro de uma detecção está dentro de uma caixa de estrutura."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa tracking simples da bola em blocos anotados.")
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--max-distance", type=float, default=300.0)
    parser.add_argument("--min-direction-cosine", type=float, default=-0.2)
    parser.add_argument("--max-area-ratio", type=float, default=4.0)
    parser.add_argument("--rejection-grace", type=int, default=1)
    parser.add_argument("--appearance-threshold", type=float, default=0.70)
    parser.add_argument(
        "--goal-model",
        type=Path,
        default=None,
        help="Modelo de estruturas de gol usado somente para filtrar o início de novos tracks.",
    )
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    parser.add_argument("--goal-imgsz", type=int, default=640)
    parser.add_argument(
        "--exclude-goal-seeds",
        action="store_true",
        help="Não inicia um novo track em uma bola cujo centro esteja dentro de uma estrutura de gol.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    blocks = read_csv(args.blocks)
    annotations = [row for row in read_csv(args.annotations) if row.get("annotation_status") == "completed"]
    block_ids = {row["block_id"] for row in blocks}
    annotations = [row for row in annotations if row.get("block_id") in block_ids]
    model = YOLO(str(args.model))
    if args.exclude_goal_seeds and args.goal_model is None:
        parser.error("--exclude-goal-seeds requer --goal-model.")
    goal_model = YOLO(str(args.goal_model)) if args.goal_model else None
    args.output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    detections_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()

    for block_index, block in enumerate(blocks, start=1):
        video_path = args.videos_dir / block["video_filename"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_step = max(1, round(video_fps / args.sample_fps))
        timestamp = float(block["start_seconds"])
        end_seconds = float(block["end_seconds"])
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        previous_center: tuple[float, float] | None = None
        velocity = (0.0, 0.0)
        last_box: tuple[float, float, float, float] | None = None
        gap = args.max_gap + 1
        rejection_streak = 0
        print(f"[{block_index}/{len(blocks)}] processando {block['block_id']} ({video_path.name})")
        while timestamp <= end_seconds + 1e-6:
            ok, image = cap.read()
            if not ok or image is None:
                break
            result = model.predict(source=image, imgsz=args.imgsz, conf=args.confidence, device=args.device, verbose=False)[0]
            candidates = [tuple(float(value) for value in box) for box in result.boxes.xyxy.tolist()]
            confidences = [float(value) for value in result.boxes.conf.tolist()]
            goal_boxes: list[tuple[float, float, float, float]] = []
            if goal_model is not None and previous_center is None and candidates:
                goal_result = goal_model.predict(
                    source=image,
                    imgsz=args.goal_imgsz,
                    conf=args.goal_confidence,
                    device=args.device,
                    verbose=False,
                )[0]
                goal_boxes = [tuple(float(value) for value in box) for box in goal_result.boxes.xyxy.tolist()]
            predicted_center = (previous_center[0] + velocity[0], previous_center[1] + velocity[1]) if previous_center else None
            chosen_index = -1
            association_rejected = False
            excluded_seed_candidates = 0
            if candidates:
                if predicted_center is not None and gap <= args.max_gap:
                    distances = [distance(center(box), predicted_center) for box in candidates]
                    valid: list[int] = []
                    previous_movement = velocity
                    movement_norm = math.hypot(*previous_movement)
                    previous_area = ((last_box[2] - last_box[0]) * (last_box[3] - last_box[1])) if last_box else 0.0
                    for candidate_index, candidate in enumerate(candidates):
                        if distances[candidate_index] > args.max_distance:
                            continue
                        if previous_area:
                            candidate_area = max(1.0, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
                            ratio = max(candidate_area / previous_area, previous_area / candidate_area)
                            if ratio > args.max_area_ratio:
                                continue
                        if movement_norm > 10.0:
                            movement = (center(candidate)[0] - previous_center[0], center(candidate)[1] - previous_center[1])
                            movement_candidate_norm = math.hypot(*movement)
                            if movement_candidate_norm > 10.0:
                                cosine = (movement[0] * previous_movement[0] + movement[1] * previous_movement[1]) / (movement_candidate_norm * movement_norm)
                                if cosine < args.min_direction_cosine:
                                    continue
                        valid.append(candidate_index)
                    if valid:
                        chosen_index = min(valid, key=distances.__getitem__)
                    else:
                        association_rejected = True
                if chosen_index < 0 and previous_center is None:
                    eligible = list(range(len(candidates)))
                    if args.exclude_goal_seeds:
                        eligible = [
                            candidate_index
                            for candidate_index in eligible
                            if not any(point_inside_box(center(candidates[candidate_index]), goal_box) for goal_box in goal_boxes)
                        ]
                        excluded_seed_candidates = len(candidates) - len(eligible)
                    if eligible:
                        chosen_index = max(eligible, key=confidences.__getitem__)
            detected = chosen_index >= 0
            if detected:
                chosen_box = candidates[chosen_index]
                chosen_center = center(chosen_box)
                if previous_center is not None:
                    velocity = (chosen_center[0] - previous_center[0], chosen_center[1] - previous_center[1])
                previous_center = chosen_center
                last_box = chosen_box
                gap = 0
                rejection_streak = 0
            elif previous_center is not None and gap < args.max_gap and (not association_rejected or rejection_streak < args.rejection_grace):
                previous_center = (previous_center[0] + velocity[0], previous_center[1] + velocity[1])
                gap += 1
                if association_rejected:
                    rejection_streak += 1
            else:
                previous_center = None
                last_box = None
                velocity = (0.0, 0.0)
                gap = args.max_gap + 1
                rejection_streak = 0
            active = previous_center is not None
            row = {
                "block_id": block["block_id"], "match_id": block["match_id"],
                "video_filename": block["video_filename"], "capture_domain": block["capture_domain"],
                "timestamp_seconds": round(timestamp, 3), "timestamp_hhmmss": format_time(timestamp),
                "detected": detected, "track_active": active, "gap": gap if active else "",
                "confidence": round(confidences[chosen_index], 6) if detected else 0.0,
                "prediction_count": len(candidates),
                "goal_seed_candidates_excluded": excluded_seed_candidates,
                "track_x1": round(last_box[0], 2) if last_box and active else "",
                "track_y1": round(last_box[1], 2) if last_box and active else "",
                "track_x2": round(last_box[2], 2) if last_box and active else "",
                "track_y2": round(last_box[3], 2) if last_box and active else "",
            }
            frame_rows.append(row)
            detections_by_block[block["block_id"]].append(row)
            for _ in range(frame_step - 1):
                if not cap.grab():
                    break
            timestamp += 1.0 / args.sample_fps
        cap.release()

    interval_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        rows = [row for row in detections_by_block[annotation["block_id"]] if float(annotation["start_seconds"]) - 1e-6 <= row["timestamp_seconds"] <= float(annotation["end_seconds"]) + 1e-6]
        active_count = sum(bool(row["track_active"]) for row in rows)
        appearance_rate = active_count / len(rows) if rows else 0.0
        confirmed = appearance_rate >= args.appearance_threshold
        interval_rows.append({
            "interval_id": annotation["interval_id"], "block_id": annotation["block_id"],
            "match_id": annotation["match_id"], "video_filename": annotation["video_filename"],
            "start_seconds": annotation["start_seconds"], "end_seconds": annotation["end_seconds"],
            "interval_label": annotation["interval_label"], "frames_analyzed": len(rows),
            "active_frames": active_count, "appearance_rate": round(appearance_rate, 6),
            "confirmed": confirmed, "result": classify(annotation["interval_label"], confirmed),
        })
    evaluable = [row for row in interval_rows if row["result"] != "inconclusive"]
    summary = {
        "experiment_id": args.output.name, "model": str(args.model),
        "blocks_total": len(blocks), "intervals_total": len(interval_rows),
        "intervals_evaluable": len(evaluable),
        "true_positive_intervals": sum(row["result"] == "true_positive" for row in evaluable),
        "true_negative_intervals": sum(row["result"] == "true_negative" for row in evaluable),
        "false_positive_intervals": sum(row["result"] == "false_positive" for row in evaluable),
        "false_negative_intervals": sum(row["result"] == "false_negative" for row in evaluable),
        "inconclusive": sum(row["result"] == "inconclusive" for row in interval_rows),
        "confidence": args.confidence, "sample_fps": args.sample_fps,
        "max_gap": args.max_gap, "max_distance": args.max_distance,
        "appearance_threshold": args.appearance_threshold,
        "execution_seconds": round(time.perf_counter() - started, 3), "status": "completed",
    }
    write_csv(args.output / "frame_tracking.csv", frame_rows)
    write_csv(args.output / "interval_tracking_results.csv", interval_rows)
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
