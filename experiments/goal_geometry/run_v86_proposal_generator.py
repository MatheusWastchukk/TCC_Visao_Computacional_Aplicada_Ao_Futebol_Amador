"""V8.6 spatial proposal generator without learned candidate ranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from experiments.goal_geometry.run_geometry_versions import (
    Segment,
    apply_exclusion_zones,
    candidate_from_lines,
    frame_path,
    make_segments,
    merge_segments,
    morphology,
    threshold_mask,
)
from experiments.goal_geometry.run_v85_candidate_level import bbox, bbox_iou, candidate_box
from experiments.goal_geometry.run_v85_learned_geometry import candidate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o gerador espacial de propostas da V8.6.")
    parser.add_argument("--development-annotations", type=Path, required=True)
    parser.add_argument("--external-annotations", type=Path, required=True)
    parser.add_argument("--external-timestamps", type=Path, required=True)
    parser.add_argument("--external-frames-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dimension", type=int, default=1920)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize_segment(segment: Segment, orientation: str) -> Segment:
    points = [(segment.x1, segment.y1), (segment.x2, segment.y2)]
    if orientation == "vertical":
        top, bottom = sorted(points, key=lambda point: point[1])
        return Segment(top[0], top[1], bottom[0], bottom[1], segment.angle, segment.length)
    left, right = sorted(points, key=lambda point: point[0])
    return Segment(left[0], left[1], right[0], right[1], segment.angle, segment.length)


def segment_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def appearance_features(
    image: np.ndarray, candidate: Any
) -> dict[str, float]:
    left, top, right, bottom = candidate_box(candidate)
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(left)))
    y1 = max(0, min(height - 1, int(top)))
    x2 = max(x1 + 1, min(width, int(right) + 1))
    y2 = max(y1 + 1, min(height, int(bottom) + 1))
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]
    upper = crop[: max(1, crop.shape[0] // 2)]
    upper_hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    upper_blue = np.logical_and(
        np.logical_and(upper_hsv[:, :, 0] >= 90, upper_hsv[:, :, 0] <= 135),
        upper_hsv[:, :, 1] >= 40,
    )
    edges = cv2.Canny(gray, 80, 160)
    return {
        "box_brightness": float(np.mean(value) / 255.0),
        "box_white_fraction": float(np.mean((value >= 150) & (saturation <= 80))),
        "box_green_fraction": float(
            np.mean((hue >= 35) & (hue <= 95) & (saturation >= 50))
        ),
        "box_blue_fraction": float(
            np.mean((hue >= 90) & (hue <= 135) & (saturation >= 40))
        ),
        "box_edge_density": float(np.mean(edges > 0)),
        "upper_blue_fraction": float(np.mean(upper_blue)),
    }


def proposal_rows(
    image: np.ndarray,
    config: dict[str, Any],
    limits: dict[str, float],
) -> list[dict[str, Any]]:
    mask = apply_exclusion_zones(morphology(threshold_mask(image, config), config), config)
    _edges, raw_segments = make_segments(mask, config)
    verticals = [
        normalize_segment(item, "vertical")
        for item in merge_segments(raw_segments, "vertical", config)
    ]
    horizontals = [
        normalize_segment(item, "horizontal")
        for item in merge_segments(raw_segments, "horizontal", config)
    ]
    frame_height, frame_width = mask.shape[:2]
    rows_out: list[dict[str, Any]] = []
    topology = config.get("topology", {})
    topology_mode = str(topology.get("mode", "hard")).lower()
    min_endpoint_tolerance = float(topology.get("min_endpoint_tolerance", 20.0))
    endpoint_tolerance_ratio = float(topology.get("endpoint_tolerance_ratio", 0.20))
    top_alignment_ratio = float(topology.get("top_alignment_ratio", 0.50))
    min_post_drop_ratio = float(topology.get("min_post_drop_ratio", 0.25))
    max_bar_slope_ratio = float(topology.get("max_bar_slope_ratio", 0.20))
    for left_index, left in enumerate(verticals):
        for right_index, right in enumerate(verticals[left_index + 1 :], start=left_index + 1):
            if left.x1 >= right.x1:
                continue
            for bar_index, bar in enumerate(horizontals):
                span = right.x1 - left.x1
                tolerance = max(min_endpoint_tolerance, endpoint_tolerance_ratio * span)
                left_top = (left.x1, left.y1)
                right_top = (right.x1, right.y1)
                bar_left = (bar.x1, bar.y1)
                bar_right = (bar.x2, bar.y2)
                left_distance = segment_distance(left_top, bar_left)
                right_distance = segment_distance(right_top, bar_right)
                endpoint_score = max(
                    0.0,
                    1.0 - (left_distance + right_distance) / (2.0 * tolerance),
                )
                if topology_mode != "soft" and (
                    left_distance > tolerance or right_distance > tolerance
                ):
                    continue
                bar_y = (bar.y1 + bar.y2) / 2.0
                left_alignment_limit = top_alignment_ratio * max(left.length, 1.0)
                right_alignment_limit = top_alignment_ratio * max(right.length, 1.0)
                left_alignment_score = max(
                    0.0, 1.0 - abs(left.y1 - bar_y) / max(left_alignment_limit, 1.0)
                )
                right_alignment_score = max(
                    0.0, 1.0 - abs(right.y1 - bar_y) / max(right_alignment_limit, 1.0)
                )
                if topology_mode != "soft" and abs(left.y1 - bar_y) > left_alignment_limit:
                    continue
                if topology_mode != "soft" and abs(right.y1 - bar_y) > right_alignment_limit:
                    continue
                left_drop_limit = min_post_drop_ratio * max(left.length, 1.0)
                right_drop_limit = min_post_drop_ratio * max(right.length, 1.0)
                left_drop_score = max(0.0, min(1.0, (left.y2 - bar_y) / max(left_drop_limit, 1.0)))
                right_drop_score = max(
                    0.0, min(1.0, (right.y2 - bar_y) / max(right_drop_limit, 1.0))
                )
                if topology_mode != "soft" and left.y2 - bar_y < left_drop_limit:
                    continue
                if topology_mode != "soft" and right.y2 - bar_y < right_drop_limit:
                    continue
                span_score = max(
                    0.0,
                    1.0
                    - (
                        max(0.0, bar.x1 - left.x1 - tolerance)
                        + max(0.0, right.x1 - tolerance - bar.x2)
                    )
                    / max(span, 1.0),
                )
                if topology_mode != "soft" and (
                    bar.x1 > left.x1 + tolerance or bar.x2 < right.x1 - tolerance
                ):
                    continue
                slope_limit = max_bar_slope_ratio * max(span, 1.0)
                slope_score = max(0.0, 1.0 - abs(bar.y2 - bar.y1) / max(slope_limit, 1.0))
                if topology_mode != "soft" and abs(bar.y2 - bar.y1) > slope_limit:
                    continue
                if left.length <= 0 or right.length <= 0 or bar.length <= 0:
                    continue
                candidate = candidate_from_lines(left, right, bar, mask, 7, config)
                if candidate is None:
                    continue
                relative_width = candidate.width / frame_width
                relative_height = candidate.height / frame_height
                relative_area = relative_width * relative_height
                if not (
                    limits["relative_width_min"] <= relative_width <= limits["relative_width_max"]
                    and limits["relative_height_min"]
                    <= relative_height
                    <= limits["relative_height_max"]
                    and limits["relative_area_min"] <= relative_area <= limits["relative_area_max"]
                ):
                    continue
                topology_score = float(
                    np.mean(
                        [
                            endpoint_score,
                            left_alignment_score,
                            right_alignment_score,
                            left_drop_score,
                            right_drop_score,
                            span_score,
                            slope_score,
                        ]
                    )
                )
                connection_score = endpoint_score
                proposal_score = (
                    0.30 * topology_score
                    + 0.25 * connection_score
                    + 0.15 * candidate.height_score
                    + 0.12 * candidate.angle_score
                    + 0.08 * candidate.white_support_score
                    + 0.10 * math.exp(-abs(math.log(max(candidate.width_height_ratio, 1e-6) / 3.0)))
                )
                rows_out.append(
                    {
                        "candidate_id": f"L{left_index:03d}_R{right_index:03d}_H{bar_index:03d}",
                        "candidate": candidate,
                        "proposal_score": proposal_score,
                        "line_left_id": f"VL{left_index:03d}",
                        "line_right_id": f"VR{right_index:03d}",
                        "line_bar_id": f"H{bar_index:03d}",
                        "left_distance": left_distance,
                        "right_distance": right_distance,
                        "topology_score": topology_score,
                        "features": {
                            **candidate_features(candidate, frame_width, frame_height),
                            **appearance_features(image, candidate),
                            "topology_score": topology_score,
                        },
                    }
                )
    return rows_out


def limits_from_annotations(annotations: list[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = {
        key: [] for key in ["relative_width", "relative_height", "relative_area"]
    }
    for row in annotations:
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        width, height = image.shape[1], image.shape[0]
        truth = bbox(
            [
                (float(row[x]), float(row[y]))
                for x, y in [("lt_x", "lt_y"), ("rt_x", "rt_y"), ("lb_x", "lb_y"), ("rb_x", "rb_y")]
            ]
        )
        relative_width = (truth[2] - truth[0]) / width
        relative_height = (truth[3] - truth[1]) / height
        values["relative_width"].append(relative_width)
        values["relative_height"].append(relative_height)
        values["relative_area"].append(relative_width * relative_height)
    limits: dict[str, float] = {}
    for key, items in values.items():
        lower, upper = np.percentile(items, [1, 99])
        margin = max((upper - lower) * 0.50, 0.01)
        limits[f"{key}_min"] = max(0.0, float(lower - margin))
        limits[f"{key}_max"] = float(upper + margin)
    return limits


def select_proposals(
    proposals: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    selection = config.get("proposal_selection", {})
    max_candidates = int(selection.get("max_candidates", 50))
    nms_iou_threshold = float(selection.get("nms_iou_threshold", 0.70))
    ordered = sorted(proposals, key=lambda item: item["proposal_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for proposal in ordered:
        proposal_box = candidate_box(proposal["candidate"])
        if any(
            bbox_iou(proposal_box, candidate_box(item["candidate"])) >= nms_iou_threshold
            for item in selected
        ):
            continue
        selected.append(proposal)
        if len(selected) >= max_candidates:
            break
    return selected


def best_iou_for(
    proposals: list[dict[str, Any]], truths: list[tuple[float, float, float, float]]
) -> float:
    return max(
        (
            bbox_iou(truth, candidate_box(item["candidate"]))
            for item in proposals
            for truth in truths
        ),
        default=0.0,
    )


def process_frame(
    image: np.ndarray,
    config: dict[str, Any],
    limits: dict[str, float],
    truths: list[tuple[float, float, float, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_proposals = proposal_rows(image, config, limits)
    proposals = select_proposals(raw_proposals, config)
    best_proposal = proposals[0] if proposals else None
    best_iou = best_iou_for(raw_proposals, truths)
    top1_iou = best_iou_for(proposals[:1], truths)
    top5_iou = best_iou_for(proposals[:5], truths)
    top10_iou = best_iou_for(proposals[:10], truths)
    best_iou_row = max(
        raw_proposals,
        key=lambda item: max(
            (bbox_iou(truth, candidate_box(item["candidate"])) for truth in truths),
            default=-1.0,
        ),
        default=None,
    )
    selected_iou = (
        max(
            (bbox_iou(truth, candidate_box(best_proposal["candidate"])) for truth in truths),
            default=0.0,
        )
        if best_proposal
        else 0.0
    )
    summary = {
        "num_candidates_raw": len(raw_proposals),
        "num_candidates": len(proposals),
        "best_iou": best_iou,
        "best_candidate_id": best_iou_row["candidate_id"] if best_iou_row else "",
        "selected_candidate_id": best_proposal["candidate_id"] if best_proposal else "",
        "selected_iou": selected_iou,
        "top1_iou": top1_iou,
        "top5_iou": top5_iou,
        "top10_iou": top10_iou,
        "found_iou_ge_0_30": best_iou >= 0.30,
        "found_iou_ge_0_50": best_iou >= 0.50,
        "top1_iou_ge_0_30": top1_iou >= 0.30,
        "top5_iou_ge_0_30": top5_iou >= 0.30,
        "top10_iou_ge_0_30": top10_iou >= 0.30,
        "top1_iou_ge_0_50": top1_iou >= 0.50,
        "top5_iou_ge_0_50": top5_iou >= 0.50,
        "top10_iou_ge_0_50": top10_iou >= 0.50,
    }
    return proposals, summary


def draw_overlay(
    image: np.ndarray,
    proposals: list[dict[str, Any]],
    summary: dict[str, Any],
    truths: list[tuple[float, float, float, float]],
    frame_id: str,
) -> np.ndarray:
    output = image.copy()
    for truth in truths:
        cv2.rectangle(
            output, (int(truth[0]), int(truth[1])), (int(truth[2]), int(truth[3])), (0, 0, 255), 3
        )
    selected = next(
        (item for item in proposals if item["candidate_id"] == summary["selected_candidate_id"]),
        None,
    )
    if selected:
        candidate = selected["candidate"]
        for segment, color, label in [
            (candidate.left_post, (255, 0, 0), selected["line_left_id"]),
            (candidate.right_post, (0, 255, 0), selected["line_right_id"]),
            (candidate.crossbar, (255, 255, 0), selected["line_bar_id"]),
        ]:
            cv2.line(
                output,
                (int(segment.x1), int(segment.y1)),
                (int(segment.x2), int(segment.y2)),
                color,
                4,
            )
            cv2.putText(
                output,
                label,
                (int(segment.x1), int(segment.y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        candidate_box_values = candidate_box(candidate)
        cv2.rectangle(
            output,
            (int(candidate_box_values[0]), int(candidate_box_values[1])),
            (int(candidate_box_values[2]), int(candidate_box_values[3])),
            (255, 0, 255),
            3,
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"V8.6 {frame_id} candidato selecionado={summary['selected_candidate_id']}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        (
            f"candidatos={summary['num_candidates']} "
            f"melhor_iou={summary['best_iou']:.3f} "
            f"iou_selecionado={summary['selected_iou']:.3f}"
        ),
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return output


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    development = rows(args.development_annotations)
    external = rows(args.external_annotations)
    timestamps = {
        row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}": row
        for row in rows(args.external_timestamps)
    }
    limits = limits_from_annotations(development)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "proposal_limits.json").write_text(
        json.dumps(limits, indent=2), encoding="utf-8"
    )
    grouped_development: dict[str, list[dict[str, str]]] = {}
    grouped_external: dict[str, list[dict[str, str]]] = {}
    for row in development:
        grouped_development.setdefault(row["frame_id"], []).append(row)
    for row in external:
        grouped_external.setdefault(row["frame_id"], []).append(row)
    grouped_inputs = [(group, "development", None) for group in grouped_development.values()]
    grouped_inputs.extend(
        (group, "external", args.external_frames_dir) for group in grouped_external.values()
    )
    all_summaries: list[dict[str, Any]] = []
    for annotation_group, source, _frames_dir in grouped_inputs:
        annotation = annotation_group[0]
        if source == "development":
            image_path = Path(annotation["image_path"])
        else:
            image_path = frame_path(args.external_frames_dir, timestamps[annotation["frame_id"]])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        scale = min(1.0, args.max_dimension / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        truths = [
            bbox(
                [
                    (float(row[x]) * scale, float(row[y]) * scale)
                    for x, y in [
                        ("lt_x", "lt_y"),
                        ("rt_x", "rt_y"),
                        ("lb_x", "lb_y"),
                        ("rb_x", "rb_y"),
                    ]
                ]
            )
            for row in annotation_group
        ]
        proposals, summary = process_frame(image, config, limits, truths)
        summary.update(
            {
                "frame_id": annotation["frame_id"],
                "match_id": annotation["match_id"],
                "source": source,
            }
        )
        all_summaries.append(summary)
        if source == "external":
            overlay = draw_overlay(image, proposals, summary, truths, annotation["frame_id"])
            overlay_dir = args.output / "overlays" / annotation["match_id"]
            overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_dir / f"{annotation['frame_id']}.jpg"), overlay)
    fields = list(all_summaries[0])
    with (args.output / "proposal_recall.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summaries)
    metrics: dict[str, Any] = {
        "experiment_id": args.output.name,
        "version": "V8.6",
        "limits": limits,
    }
    for source in ["development", "external"]:
        selected = [row for row in all_summaries if row["source"] == source]
        metrics[source] = {
            "frames": len(selected),
            "proposal_recall_iou_030": sum(row["found_iou_ge_0_30"] for row in selected)
            / len(selected),
            "proposal_recall_iou_050": sum(row["found_iou_ge_0_50"] for row in selected)
            / len(selected),
            "top1_recall_iou_030": sum(row["top1_iou_ge_0_30"] for row in selected)
            / len(selected),
            "top5_recall_iou_030": sum(row["top5_iou_ge_0_30"] for row in selected)
            / len(selected),
            "top10_recall_iou_030": sum(row["top10_iou_ge_0_30"] for row in selected)
            / len(selected),
            "top1_recall_iou_050": sum(row["top1_iou_ge_0_50"] for row in selected)
            / len(selected),
            "top5_recall_iou_050": sum(row["top5_iou_ge_0_50"] for row in selected)
            / len(selected),
            "top10_recall_iou_050": sum(row["top10_iou_ge_0_50"] for row in selected)
            / len(selected),
            "mean_candidates_raw": float(np.mean([row["num_candidates_raw"] for row in selected])),
            "mean_candidates": float(np.mean([row["num_candidates"] for row in selected])),
            "mean_best_iou": float(np.mean([row["best_iou"] for row in selected])),
            "mean_top1_iou": float(np.mean([row["top1_iou"] for row in selected])),
            "mean_top5_iou": float(np.mean([row["top5_iou"] for row in selected])),
        }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
