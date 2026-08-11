"""Prepara recortes de imagens e rótulos para o experimento híbrido curto da V8.7."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from experiments.goal_geometry.run_v85_candidate_level import bbox, bbox_iou, candidate_box
from experiments.goal_geometry.run_v86_proposal_generator import (
    limits_from_annotations,
    proposal_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara os recortes de propostas da V8.7.")
    parser.add_argument("--development-annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-positive-crops", type=int, default=20)
    parser.add_argument("--max-negative-crops", type=int, default=30)
    parser.add_argument("--crop-size", type=int, default=224)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def truth_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return bbox(
        [
            (float(row[x]), float(row[y]))
            for x, y in [("lt_x", "lt_y"), ("rt_x", "rt_y"), ("lb_x", "lb_y"), ("rb_x", "rb_y")]
        ]
    )


def grouped(annotation_rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in annotation_rows:
        groups.setdefault(row["frame_id"], []).append(row)
    return list(groups.values())


def crop_with_padding(
    image: np.ndarray, box: tuple[float, float, float, float], size: int
) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = box
    box_width = max(right - left, 1.0)
    box_height = max(bottom - top, 1.0)
    padding_x = 0.10 * box_width
    padding_y = 0.10 * box_height
    x1 = max(0, int(left - padding_x))
    y1 = max(0, int(top - padding_y))
    x2 = min(width, max(x1 + 1, int(right + padding_x)))
    y2 = min(height, max(y1 + 1, int(bottom + padding_y)))
    crop = image[y1:y2, x1:x2]
    scale = min(size / crop.shape[1], size / crop.shape[0])
    resized_width = max(1, int(round(crop.shape[1] * scale)))
    resized_height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def intersection_over_truth(
    candidate: tuple[float, float, float, float],
    truth: tuple[float, float, float, float],
) -> float:
    left = max(candidate[0], truth[0])
    top = max(candidate[1], truth[1])
    right = min(candidate[2], truth[2])
    bottom = min(candidate[3], truth[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    truth_area = max(0.0, truth[2] - truth[0]) * max(0.0, truth[3] - truth[1])
    return intersection / truth_area if truth_area else 0.0


def negative_category(features: dict[str, float]) -> str:
    if features["upper_blue_fraction"] >= 0.30 or features["box_blue_fraction"] >= 0.35:
        return "sky_like"
    if features["box_green_fraction"] >= 0.35:
        return "field_like"
    if features["box_edge_density"] >= 0.20:
        return "scene_or_player_like"
    return "geometry_false_positive"


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    annotations = rows(args.development_annotations)
    if not annotations or any(row["annotation_status"] != "completed" for row in annotations):
        raise ValueError("As anotações de desenvolvimento precisam estar completas")
    limits = limits_from_annotations(annotations)
    if args.output.exists():
        shutil.rmtree(args.output)
    crop_root = args.output / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    discarded_negative_goal_overlap = 0
    for group in grouped(annotations):
        annotation = group[0]
        image = cv2.imread(annotation["image_path"])
        if image is None:
            raise FileNotFoundError(annotation["image_path"])
        truths = [truth_box(row) for row in group]
        proposals = proposal_rows(image, config, limits)
        scored: list[tuple[float, dict[str, Any], float]] = []
        for proposal in proposals:
            proposal_iou = max(
                (bbox_iou(truth, candidate_box(proposal["candidate"])) for truth in truths),
                default=0.0,
            )
            scored.append((proposal_iou, proposal, proposal_iou))
        positives = sorted(
            [item for item in scored if item[0] >= 0.30],
            key=lambda item: item[0],
            reverse=True,
        )[: args.max_positive_crops]
        negative_candidates = []
        for item in scored:
            if item[0] >= 0.30:
                continue
            candidate_box_values = candidate_box(item[1]["candidate"])
            if any(
                intersection_over_truth(candidate_box_values, truth) >= 0.15 for truth in truths
            ):
                discarded_negative_goal_overlap += 1
                continue
            negative_candidates.append(item)
        negatives = sorted(
            negative_candidates,
            key=lambda item: item[1]["proposal_score"],
            reverse=True,
        )[: args.max_negative_crops]
        for label, selected in [("goal_structure", positives), ("not_goal", negatives)]:
            for index, (proposal_iou, proposal, _score) in enumerate(selected, start=1):
                box = candidate_box(proposal["candidate"])
                crop = crop_with_padding(image, box, args.crop_size)
                category = (
                    "goal_structure"
                    if label == "goal_structure"
                    else negative_category(proposal["features"])
                )
                filename = f"{annotation['frame_id']}_{label}_{index:03d}.jpg"
                output_dir = crop_root / label / category
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / filename
                cv2.imwrite(str(output_path), crop)
                manifest.append(
                    {
                        "crop_path": str(output_path),
                        "frame_id": annotation["frame_id"],
                        "match_id": annotation["match_id"],
                        "label": label,
                        "negative_category": category if label == "not_goal" else "",
                        "candidate_iou": proposal_iou,
                        "proposal_score": proposal["proposal_score"],
                        "topology_score": proposal["topology_score"],
                    }
                )
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "experiment_id": args.output.name,
        "version": "V8.7_crop_preparation",
        "source_frames": len(grouped(annotations)),
        "crops": len(manifest),
        "positive_crops": sum(row["label"] == "goal_structure" for row in manifest),
        "negative_crops": sum(row["label"] == "not_goal" for row in manifest),
        "discarded_negative_goal_overlap": discarded_negative_goal_overlap,
        "negative_categories": {
            category: sum(row["negative_category"] == category for row in manifest)
            for category in [
                "sky_like",
                "field_like",
                "scene_or_player_like",
                "geometry_false_positive",
            ]
        },
        "positive_rule": "candidate IoU >= 0.30",
        "negative_rule": "candidate IoU < 0.30",
        "crop_size": args.crop_size,
        "resize_mode": "letterbox_preserving_aspect_ratio",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
