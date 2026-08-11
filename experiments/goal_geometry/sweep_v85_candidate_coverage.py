"""Sweep Hough parameters using manual annotations as a development set."""

from __future__ import annotations

import argparse
import copy
import csv
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
from experiments.goal_geometry.run_v85_candidate_level import bbox, bbox_iou, candidate_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep V8.5 Hough coverage parameters.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def variants(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    settings = [
        ("gray160_none", "gray_fixed", 160, "none", 0, 15, 20, 20),
        ("gray180_close3", "gray_fixed", 180, "close", 3, 20, 25, 30),
        ("gray200_close3", "gray_fixed", 200, "close", 3, 20, 25, 30),
        ("gray180_open_close3", "gray_fixed", 180, "open_close", 3, 20, 25, 30),
        ("hsv160_close3", "hsv_white", 160, "close", 3, 20, 25, 30),
        ("hsv180_close3", "hsv_white", 180, "close", 3, 20, 25, 30),
        ("gray180_close5", "gray_fixed", 180, "close", 5, 15, 20, 40),
        ("gray180_close3_gap40", "gray_fixed", 180, "close", 3, 15, 25, 40),
        ("gray160_open_close5", "gray_fixed", 160, "open_close", 5, 15, 20, 40),
        ("gray200_open_close5", "gray_fixed", 200, "open_close", 5, 15, 20, 40),
        ("gray160_close3_len35", "gray_fixed", 160, "close", 3, 15, 35, 40),
        ("gray180_close3_len35", "gray_fixed", 180, "close", 3, 15, 35, 40),
    ]
    result = []
    for name, mode, value, operation, kernel, hough_threshold, min_length, max_gap in settings:
        config = copy.deepcopy(base)
        config["threshold"].update({"mode": mode, "value": value})
        config["morphology"].update({"operation": operation, "kernel": kernel})
        config["hough"].update(
            {
                "threshold": hough_threshold,
                "min_line_length": min_length,
                "max_line_gap": max_gap,
            }
        )
        result.append((name, config))
    return result


def main() -> int:
    args = parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    annotations = list(csv.DictReader(args.annotations.open(newline="", encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    for name, config in variants(base):
        frame_count = 0
        positive_frames = 0
        candidate_count = 0
        max_ious: list[float] = []
        for annotation in annotations:
            image = cv2.imread(annotation["image_path"])
            if image is None:
                raise FileNotFoundError(annotation["image_path"])
            scale = min(1.0, 960.0 / max(image.shape[:2]))
            resized = (
                image
                if scale == 1.0
                else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            )
            points = [
                (float(annotation[x]) * scale, float(annotation[y]) * scale)
                for x, y in [("lt_x", "lt_y"), ("rt_x", "rt_y"), ("lb_x", "lb_y"), ("rb_x", "rb_y")]
            ]
            truth = bbox(points)
            mask = morphology(threshold_mask(resized, config), config)
            _edges, segments = make_segments(mask, config)
            candidates = candidate_pool(mask, segments, config, merged=True)
            ious = [bbox_iou(truth, candidate_box(candidate)) for candidate in candidates]
            frame_count += 1
            candidate_count += len(candidates)
            max_iou = max(ious, default=0.0)
            max_ious.append(max_iou)
            if max_iou >= 0.30:
                positive_frames += 1
        rows.append(
            {
                "variant": name,
                "frames": frame_count,
                "frames_with_iou_ge_0_30": positive_frames,
                "coverage": positive_frames / frame_count if frame_count else 0.0,
                "candidates": candidate_count,
                "mean_max_iou": float(np.mean(max_ious)),
                "max_iou": max(max_ious, default=0.0),
            }
        )
    rows.sort(key=lambda row: (row["frames_with_iou_ge_0_30"], row["mean_max_iou"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
