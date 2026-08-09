"""Evaluate persistence and spatial consistency on real video sequences."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from experiments.goal_geometry.run_geometry_versions import (
    candidate_pool,
    make_segments,
    morphology,
    threshold_mask,
)
from experiments.goal_geometry.run_v8_soft_geometry import calibration, soft_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V9 temporal validation.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-threshold", type=float, default=0.30)
    parser.add_argument("--appearance-threshold", type=float, default=0.40)
    return parser.parse_args()


def longest_streak(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main() -> int:
    args = parse_args()
    sequences = list(csv.DictReader(args.sequences.open(newline="", encoding="utf-8")))
    if any(not row["start_seconds"] or not row["end_seconds"] for row in sequences):
        raise ValueError("V9 exige start_seconds e end_seconds preenchidos em todas as sequencias")
    annotations = list(csv.DictReader(args.annotations.open(newline="", encoding="utf-8")))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stats = calibration(annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for sequence in sequences:
        cap = cv2.VideoCapture(sequence["video_path"])
        if not cap.isOpened():
            raise FileNotFoundError(sequence["video_path"])
        fps = float(sequence.get("fps") or 5)
        start_seconds = float(sequence["start_seconds"])
        end_seconds = float(sequence["end_seconds"])
        step = 1.0 / fps
        scores: list[float] = []
        centers: list[tuple[float, float] | None] = []
        timestamp = start_seconds
        while timestamp <= end_seconds + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, image = cap.read()
            if not ok or image is None:
                break
            scale = min(1.0, 960.0 / max(image.shape[:2]))
            if scale < 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mask = morphology(threshold_mask(image, config), config)
            _edges, segments = make_segments(mask, config)
            candidates = candidate_pool(mask, segments, config, merged=True)
            scored = [
                (candidate, soft_score(candidate, stats)["geometry_score"])
                for candidate in candidates
            ]
            best = max(scored, key=lambda item: item[1], default=None)
            score = best[1] if best else 0.0
            center = None
            if best:
                candidate = best[0]
                center = (
                    (candidate.left_post.x1 + candidate.right_post.x1) / 2,
                    (candidate.crossbar.y1 + max(candidate.left_post.y1, candidate.right_post.y1))
                    / 2,
                )
            scores.append(score)
            centers.append(center)
            frame_rows.append(
                {
                    "sequence_id": sequence["sequence_id"],
                    "timestamp_seconds": timestamp,
                    "score": score,
                    "detected": score >= args.geometry_threshold,
                    "center_x": center[0] if center else "",
                    "center_y": center[1] if center else "",
                }
            )
            timestamp += step
        cap.release()
        detected = [
            row["detected"] for row in frame_rows if row["sequence_id"] == sequence["sequence_id"]
        ]
        valid_centers = [center for center in centers if center is not None]
        center_std = (
            float(np.mean(np.std(np.array(valid_centers), axis=0)))
            if valid_centers
            else float("inf")
        )
        appearance = sum(detected) / len(detected) if detected else 0.0
        confirmed = appearance >= args.appearance_threshold and bool(valid_centers)
        sequence_rows.append(
            {
                "sequence_id": sequence["sequence_id"],
                "match_id": sequence["match_id"],
                "true_label": sequence["true_label"],
                "frames_analyzed": len(detected),
                "candidate_frames": sum(detected),
                "appearance_rate": appearance,
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "max_score": max(scores, default=0.0),
                "longest_streak": longest_streak(detected),
                "center_std": center_std,
                "confirmed": confirmed,
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
    report = {
        "experiment_id": "EXP-0022",
        "version": "V9",
        "sequences": len(sequence_rows),
        "appearance_threshold": args.appearance_threshold,
        "geometry_threshold": args.geometry_threshold,
        "execution_seconds": time.perf_counter() - start_time,
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
