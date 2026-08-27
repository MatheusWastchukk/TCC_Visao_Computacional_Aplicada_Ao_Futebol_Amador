"""Executa o piloto temporal usando o detector espacial V10 congelado."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o piloto temporal do detector V10.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--appearance-threshold", type=float, default=0.70)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequences = list(csv.DictReader(args.sequences.open(newline="", encoding="utf-8")))
    if any(not row["start_seconds"] or not row["end_seconds"] for row in sequences):
        raise ValueError("Todas as sequências precisam de start_seconds e end_seconds.")
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for sequence in sequences:
        cap = cv2.VideoCapture(sequence["video_path"])
        if not cap.isOpened():
            raise FileNotFoundError(sequence["video_path"])
        fps = float(sequence.get("fps") or 5)
        timestamp = float(sequence["start_seconds"])
        end_seconds = float(sequence["end_seconds"])
        detections: list[bool] = []
        centers: list[tuple[float, float]] = []
        scores: list[float] = []
        while timestamp <= end_seconds + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
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
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
            confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
            detected = len(boxes) > 0
            detections.append(detected)
            scores.append(float(confidences.max()) if len(confidences) else 0.0)
            if detected:
                best_index = int(confidences.argmax())
                best_box = boxes[best_index]
                centers.append(
                    (float((best_box[0] + best_box[2]) / 2), float((best_box[1] + best_box[3]) / 2))
                )
            frame_rows.append(
                {
                    "sequence_id": sequence["sequence_id"],
                    "timestamp_seconds": round(timestamp, 3),
                    "detected": detected,
                    "confidence": round(scores[-1], 6),
                    "prediction_count": len(boxes),
                }
            )
            timestamp += 1.0 / fps
        cap.release()
        appearance_rate = sum(detections) / len(detections) if detections else 0.0
        confirmed = appearance_rate >= args.appearance_threshold
        center_std = float(np.mean(np.std(np.asarray(centers), axis=0))) if centers else None
        label = sequence["true_label"]
        sequence_rows.append(
            {
                "sequence_id": sequence["sequence_id"],
                "match_id": sequence["match_id"],
                "true_label": label,
                "frames_analyzed": len(detections),
                "candidate_frames": sum(detections),
                "appearance_rate": round(appearance_rate, 6),
                "mean_confidence": round(float(np.mean(scores)), 6) if scores else 0.0,
                "center_std": round(center_std, 6) if center_std is not None else "",
                "confirmed": confirmed,
                "evaluable": label in {"goal_structure", "not_goal"},
                "result": (
                    "inconclusive"
                    if label == "inconclusive"
                    else "true_positive"
                    if label == "goal_structure" and confirmed
                    else "false_negative"
                    if label == "goal_structure"
                    else "false_positive"
                    if confirmed
                    else "true_negative"
                ),
            }
        )

    with (args.output / "frame_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    with (args.output / "temporal_sequences.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)

    evaluable = [row for row in sequence_rows if row["evaluable"]]
    total_seconds = sum(float(row["end_seconds"]) - float(row["start_seconds"]) for row in sequences)
    summary = {
        "experiment_id": "EXP-0092-TEMPORAL-V1",
        "version": "V10_temporal",
        "sequences_total": len(sequence_rows),
        "sequences_evaluable": len(evaluable),
        "sequences_inconclusive": sum(row["true_label"] == "inconclusive" for row in sequence_rows),
        "true_positive_sequences": sum(row["result"] == "true_positive" for row in evaluable),
        "true_negative_sequences": sum(row["result"] == "true_negative" for row in evaluable),
        "false_positive_sequences": sum(row["result"] == "false_positive" for row in evaluable),
        "false_negative_sequences": sum(row["result"] == "false_negative" for row in evaluable),
        "appearance_threshold": args.appearance_threshold,
        "confidence": args.confidence,
        "total_sequence_minutes": round(total_seconds / 60, 6),
        "false_candidates_per_hour": round(
            sum(row["result"] == "false_positive" for row in evaluable) / (total_seconds / 3600), 6
        )
        if total_seconds
        else None,
        "execution_seconds": round(time.perf_counter() - start_time, 3),
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
