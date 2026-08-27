"""Executa a validação temporal V2 em blocos contínuos anotados."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o detector V10 em blocos contínuos anotados."
    )
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--appearance-threshold", type=float, default=0.70)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    blocks = read_csv(args.blocks)
    annotations = [
        row
        for row in read_csv(args.annotations)
        if row.get("annotation_status") == "completed"
    ]
    if not blocks or not annotations:
        raise ValueError("Blocos e intervalos anotados são obrigatórios.")

    block_ids = {row["block_id"] for row in blocks}
    annotations = [row for row in annotations if row.get("block_id") in block_ids]
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    detections_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
        print(f"[{block_index}/{len(blocks)}] processando {block['block_id']} ({video_path.name})")
        while timestamp <= end_seconds + 1e-6:
            ok, image = cap.read()
            if not ok or image is None:
                break
            results = model.predict(
                source=image,
                imgsz=args.imgsz,
                conf=args.confidence,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            boxes = (
                result.boxes.xyxy.cpu().numpy()
                if result.boxes is not None
                else np.empty((0, 4))
            )
            confidences = (
                result.boxes.conf.cpu().numpy()
                if result.boxes is not None
                else np.empty(0)
            )
            row = {
                "block_id": block["block_id"],
                "match_id": block["match_id"],
                "video_filename": block["video_filename"],
                "capture_domain": block["capture_domain"],
                "timestamp_seconds": round(timestamp, 3),
                "timestamp_hhmmss": format_time(timestamp),
                "detected": bool(len(boxes)),
                "confidence": round(float(confidences.max()) if len(confidences) else 0.0, 6),
                "prediction_count": len(boxes),
            }
            frame_rows.append(row)
            detections_by_block[block["block_id"]].append(row)
            for _ in range(frame_step - 1):
                if not cap.grab():
                    break
            timestamp += 1.0 / args.sample_fps
        cap.release()

    annotations_by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_block[annotation["block_id"]].append(annotation)

    for annotation in annotations:
        start = float(annotation["start_seconds"])
        end = float(annotation["end_seconds"])
        rows = [
            row
            for row in detections_by_block[annotation["block_id"]]
            if start - 1e-6 <= row["timestamp_seconds"] <= end + 1e-6
        ]
        detected_count = sum(bool(row["detected"]) for row in rows)
        appearance_rate = detected_count / len(rows) if rows else 0.0
        confirmed = appearance_rate >= args.appearance_threshold
        interval_rows.append(
            {
                "interval_id": annotation["interval_id"],
                "block_id": annotation["block_id"],
                "match_id": annotation["match_id"],
                "video_filename": annotation["video_filename"],
                "capture_domain": next(
                    block["capture_domain"] for block in blocks if block["block_id"] == annotation["block_id"]
                ),
                "start_seconds": annotation["start_seconds"],
                "end_seconds": annotation["end_seconds"],
                "duration_seconds": round(end - start, 3),
                "interval_label": annotation["interval_label"],
                "structure_scope": annotation["structure_scope"],
                "frames_analyzed": len(rows),
                "candidate_frames": detected_count,
                "appearance_rate": round(appearance_rate, 6),
                "confirmed": confirmed,
                "result": classify(annotation["interval_label"], confirmed),
            }
        )

    evaluable = [row for row in interval_rows if row["result"] != "inconclusive"]
    block_summaries: list[dict[str, Any]] = []
    for block in blocks:
        rows = [row for row in interval_rows if row["block_id"] == block["block_id"]]
        block_summaries.append(
            {
                "block_id": block["block_id"],
                "match_id": block["match_id"],
                "capture_domain": block["capture_domain"],
                "intervals": len(rows),
                "evaluable_intervals": sum(row["result"] != "inconclusive" for row in rows),
                "true_positive": sum(row["result"] == "true_positive" for row in rows),
                "true_negative": sum(row["result"] == "true_negative" for row in rows),
                "false_positive": sum(row["result"] == "false_positive" for row in rows),
                "false_negative": sum(row["result"] == "false_negative" for row in rows),
                "inconclusive": sum(row["result"] == "inconclusive" for row in rows),
            }
        )

    write_csv(args.output / "frame_predictions.csv", frame_rows)
    write_csv(args.output / "interval_results.csv", interval_rows)
    write_csv(args.output / "block_metrics.csv", block_summaries)

    total_seconds = sum(float(block["end_seconds"]) - float(block["start_seconds"]) for block in blocks)
    negative_seconds = sum(
        float(row["duration_seconds"])
        for row in interval_rows
        if row["interval_label"] == "not_goal"
    )
    summary = {
        "experiment_id": "EXP-0092-TEMPORAL-V2",
        "version": "V10_temporal_continuous",
        "blocks_total": len(blocks),
        "intervals_total": len(interval_rows),
        "intervals_evaluable": len(evaluable),
        "intervals_inconclusive": sum(row["result"] == "inconclusive" for row in interval_rows),
        "true_positive_intervals": sum(row["result"] == "true_positive" for row in evaluable),
        "true_negative_intervals": sum(row["result"] == "true_negative" for row in evaluable),
        "false_positive_intervals": sum(row["result"] == "false_positive" for row in evaluable),
        "false_negative_intervals": sum(row["result"] == "false_negative" for row in evaluable),
        "capture_domains": sorted({block["capture_domain"] for block in blocks}),
        "confidence": args.confidence,
        "sample_fps": args.sample_fps,
        "appearance_threshold": args.appearance_threshold,
        "total_block_minutes": round(total_seconds / 60, 6),
        "negative_evaluable_minutes": round(negative_seconds / 60, 6),
        "false_candidates_per_hour_of_not_goal": round(
            sum(row["result"] == "false_positive" for row in evaluable) / (negative_seconds / 3600), 6
        )
        if negative_seconds
        else None,
        "execution_seconds": round(time.perf_counter() - start_time, 3),
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
