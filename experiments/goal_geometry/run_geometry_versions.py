"""Run the incremental Goal Geometry Detector V2 and persist every result.

The seven versions are cumulative engineering checkpoints:
V1 baseline, V2 merged segments, V3 connectivity, V4 angle/height coherence,
V5 width/height proportions, V6 white-pixel support and V7 weighted score.
Each execution writes reproducible predictions and candidate geometry details.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float
    length: float


@dataclass(frozen=True)
class Candidate:
    left_post: Segment
    right_post: Segment
    crossbar: Segment
    width: float
    height: float
    width_height_ratio: float
    relative_width: float
    relative_height: float
    height_score: float
    angle_score: float
    ratio_score: float
    connection_score: float
    white_support_left: float
    white_support_right: float
    white_support_crossbar: float
    white_support_score: float
    size_score: float
    geometry_score: float
    selected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geometry detector versions V1-V7.")
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-version", default="dataset_frames_v1")
    return parser.parse_args()


def timestamp_label(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    whole_seconds, ms = divmod(milliseconds, 1000)
    return f"{whole_seconds:06d}.{ms:03d}"


def frame_path(frames_dir: Path, row: dict[str, str]) -> Path:
    label = timestamp_label(float(row["timestamp_seconds"]))
    return frames_dir / row["label"] / f"{row['match_id']}_{label}.jpg"


def threshold_mask(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mode = config["threshold"]["mode"]
    if mode == "hsv_white":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 2] >= config["threshold"]["value"]) & (
            hsv[:, :, 1] <= config["threshold"]["max_saturation"]
        )
        return mask.astype(np.uint8) * 255
    if mode == "gray_adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    _, mask = cv2.threshold(gray, config["threshold"]["value"], 255, cv2.THRESH_BINARY)
    return mask


def morphology(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    operation = config["morphology"]["operation"]
    size = config["morphology"]["kernel"]
    if operation == "none":
        return mask
    kernel = np.ones((size, size), dtype=np.uint8)
    result = (
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if operation in {"open", "open_close"}
        else mask
    )
    if operation in {"close", "open_close"}:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def apply_exclusion_zones(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Remove configured image regions that are known non-field artifacts."""
    height, width = mask.shape[:2]
    result = mask.copy()
    for zone in config.get("exclusion_zones", []):
        x_min = max(0, min(width, int(float(zone["x_min"]) * width)))
        y_min = max(0, min(height, int(float(zone["y_min"]) * height)))
        x_max = max(x_min, min(width, int(float(zone["x_max"]) * width)))
        y_max = max(y_min, min(height, int(float(zone["y_max"]) * height)))
        result[y_min:y_max, x_min:x_max] = 0
    return result


def make_segments(mask: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, list[Segment]]:
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=config["hough"]["threshold"],
        minLineLength=config["hough"]["min_line_length"],
        maxLineGap=config["hough"]["max_line_gap"],
    )
    segments: list[Segment] = []
    if lines is None:
        return edges, segments
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = abs(math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))))
        angle = min(angle, 180.0 - angle)
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        segments.append(Segment(float(x1), float(y1), float(x2), float(y2), angle, length))
    return edges, segments


def merge_segments(
    segments: list[Segment], orientation: str, config: dict[str, Any]
) -> list[Segment]:
    if not segments:
        return []
    angle_limit = config["merge"]["angle_tolerance"]
    distance_limit = config["merge"]["distance_tolerance"]
    selected = []
    for segment in segments:
        is_vertical = (
            orientation == "vertical"
            and abs(segment.angle - 90) <= config["hough"]["vertical_tolerance"]
        )
        is_horizontal = (
            orientation == "horizontal" and segment.angle <= config["hough"]["horizontal_tolerance"]
        )
        if is_vertical or is_horizontal:
            selected.append(segment)
    merged: list[Segment] = []
    for segment in selected:
        center = (
            (segment.x1 + segment.x2) / 2
            if orientation == "vertical"
            else (segment.y1 + segment.y2) / 2
        )
        match = None
        for item in merged:
            item_center = (
                (item.x1 + item.x2) / 2 if orientation == "vertical" else (item.y1 + item.y2) / 2
            )
            if (
                abs(item.angle - segment.angle) <= angle_limit
                and abs(item_center - center) <= distance_limit
            ):
                match = item
                break
        if match is None:
            merged.append(segment)
        else:
            points = [
                (match.x1, match.y1),
                (match.x2, match.y2),
                (segment.x1, segment.y1),
                (segment.x2, segment.y2),
            ]
            if orientation == "vertical":
                x = sum(point[0] for point in points) / 4
                y_values = [point[1] for point in points]
                merged[merged.index(match)] = Segment(
                    x,
                    min(y_values),
                    x,
                    max(y_values),
                    90.0,
                    max(y_values) - min(y_values),
                )
            else:
                y = sum(point[1] for point in points) / 4
                x_values = [point[0] for point in points]
                merged[merged.index(match)] = Segment(
                    min(x_values), y, max(x_values), y, 0.0, max(x_values) - min(x_values)
                )
    return merged


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def line_support(mask: np.ndarray, segment: Segment) -> float:
    count = max(2, int(segment.length))
    xs = np.rint(np.linspace(segment.x1, segment.x2, count)).astype(int)
    ys = np.rint(np.linspace(segment.y1, segment.y2, count)).astype(int)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    return float(np.mean(mask[ys[valid], xs[valid]] > 0)) if np.any(valid) else 0.0


def connection_score(left: Segment, right: Segment, bar: Segment, tolerance: float) -> float:
    left_top = (left.x1, min(left.y1, left.y2))
    right_top = (right.x1, min(right.y1, right.y2))
    bar_left = (min(bar.x1, bar.x2), (bar.y1 + bar.y2) / 2)
    bar_right = (max(bar.x1, bar.x2), (bar.y1 + bar.y2) / 2)
    distances = [point_distance(left_top, bar_left), point_distance(right_top, bar_right)]
    return max(0.0, 1.0 - sum(min(distance / tolerance, 1.0) for distance in distances) / 2.0)


def candidate_from_lines(
    left: Segment,
    right: Segment,
    bar: Segment,
    mask: np.ndarray,
    version: int,
    config: dict[str, Any],
) -> Candidate | None:
    if left.x1 >= right.x1 or bar.x1 > bar.x2:
        return None
    width = abs(right.x1 - left.x1)
    height = (left.length + right.length) / 2
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    frame_height, frame_width = mask.shape[:2]
    relative_width = width / frame_width
    relative_height = height / frame_height
    height_score = min(left.length, right.length) / max(left.length, right.length)
    angle_score = max(
        0.0, 1.0 - abs(left.angle - right.angle) / config["geometry"]["max_angle_difference"]
    )
    ratio_score = (
        1.0 if config["geometry"]["ratio_min"] <= ratio <= config["geometry"]["ratio_max"] else 0.0
    )
    size_score = 1.0 if relative_height >= config["geometry"]["min_relative_height"] else 0.0
    connection = connection_score(
        left, right, bar, config["geometry"]["connection_tolerance"] * frame_width
    )
    support_left = line_support(mask, left)
    support_right = line_support(mask, right)
    support_bar = line_support(mask, bar)
    support_score = (support_left + support_right + support_bar) / 3.0
    if version < 3:
        connection = 1.0
    if version < 4:
        height_score = 1.0
        angle_score = 1.0
    if version < 5:
        ratio_score = 1.0
    if version < 6:
        support_score = 1.0
    score = (
        0.25 * connection
        + 0.15 * ratio_score
        + 0.10 * height_score
        + 0.10 * angle_score
        + 0.20 * support_score
        + 0.10 * size_score
        + 0.10 * min(1.0, (width + height) / (frame_width + frame_height))
    )
    selected = version >= 7 and score >= config["geometry"]["min_score"] or version < 7
    return Candidate(
        left,
        right,
        bar,
        width,
        height,
        ratio,
        relative_width,
        relative_height,
        height_score,
        angle_score,
        ratio_score,
        connection,
        support_left,
        support_right,
        support_bar,
        support_score,
        size_score,
        score,
        selected,
    )


def candidate_pool(
    mask: np.ndarray,
    segments: list[Segment],
    config: dict[str, Any],
    merged: bool,
) -> list[Candidate]:
    if merged:
        verticals = merge_segments(segments, "vertical", config)
        horizontals = merge_segments(segments, "horizontal", config)
    else:
        verticals = [
            item
            for item in segments
            if abs(item.angle - 90) <= config["hough"]["vertical_tolerance"]
        ]
        horizontals = [
            item for item in segments if item.angle <= config["hough"]["horizontal_tolerance"]
        ]
    candidates: list[Candidate] = []
    for index, left in enumerate(verticals):
        for right in verticals[index + 1 :]:
            for bar in horizontals:
                candidate = candidate_from_lines(left, right, bar, mask, 7, config)
                if candidate is None:
                    continue
                candidates.append(candidate)
    return candidates


def filter_stage(
    candidates: list[Candidate], stage: str, config: dict[str, Any]
) -> list[Candidate]:
    if stage == "hough" or stage == "merge":
        return candidates
    if stage == "connectivity":
        return [
            item
            for item in candidates
            if item.connection_score >= config["geometry"]["min_connection_score"]
        ]
    if stage == "angle_height":
        return [
            item
            for item in candidates
            if item.angle_score >= config["geometry"]["min_angle_score"]
            and item.height_score >= config["geometry"]["min_height_score"]
        ]
    if stage == "ratio":
        return [item for item in candidates if item.ratio_score >= 1.0]
    if stage == "white_support":
        return [
            item
            for item in candidates
            if item.white_support_score >= config["geometry"]["min_white_support"]
        ]
    if stage == "score":
        return [
            item for item in candidates if item.geometry_score >= config["geometry"]["min_score"]
        ]
    raise ValueError(f"Unknown geometry stage: {stage}")


def candidates_by_stage(
    mask: np.ndarray, segments: list[Segment], config: dict[str, Any]
) -> dict[str, list[Candidate]]:
    stages: dict[str, list[Candidate]] = {}
    stages["hough"] = candidate_pool(mask, segments, config, merged=False)
    stages["merge"] = candidate_pool(mask, segments, config, merged=True)
    stages["connectivity"] = filter_stage(stages["merge"], "connectivity", config)
    stages["angle_height"] = filter_stage(stages["connectivity"], "angle_height", config)
    stages["ratio"] = filter_stage(stages["angle_height"], "ratio", config)
    stages["white_support"] = filter_stage(stages["ratio"], "white_support", config)
    stages["score"] = filter_stage(stages["white_support"], "score", config)
    return stages


def find_candidates(
    mask: np.ndarray, segments: list[Segment], version: int, config: dict[str, Any]
) -> list[Candidate]:
    stages = candidates_by_stage(mask, segments, config)
    stage_names = [
        "hough",
        "merge",
        "connectivity",
        "angle_height",
        "ratio",
        "white_support",
        "score",
    ]
    selected = stages[stage_names[version - 1]]
    if version < 7:
        return selected
    return sorted(selected, key=lambda item: item.geometry_score, reverse=True)[:1]


def evaluate(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]], key: str
) -> dict[str, float | int]:
    expected = [row[key] for row in rows]
    predicted = [row["predicted"] for row in predictions]
    tp = sum(a and b for a, b in zip(expected, predicted, strict=True))
    fp = sum(not a and b for a, b in zip(expected, predicted, strict=True))
    tn = sum(not a and not b for a, b in zip(expected, predicted, strict=True))
    fn = sum(a and not b for a, b in zip(expected, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def flatten_candidate(candidate: Candidate | None) -> dict[str, Any]:
    if candidate is None:
        return {
            key: ""
            for key in [
                "left_post_x1",
                "left_post_y1",
                "left_post_x2",
                "left_post_y2",
                "right_post_x1",
                "right_post_y1",
                "right_post_x2",
                "right_post_y2",
                "crossbar_x1",
                "crossbar_y1",
                "crossbar_x2",
                "crossbar_y2",
                "width",
                "height",
                "width_height_ratio",
                "relative_width",
                "relative_height",
                "height_score",
                "angle_score",
                "ratio_score",
                "connection_score",
                "white_support_left",
                "white_support_right",
                "white_support_crossbar",
                "white_support_score",
                "size_score",
                "geometry_score",
                "selected",
            ]
        }
    values = asdict(candidate)
    for name, segment in [
        ("left_post", candidate.left_post),
        ("right_post", candidate.right_post),
        ("crossbar", candidate.crossbar),
    ]:
        values.pop(name, None)
        values[f"{name}_x1"] = segment.x1
        values[f"{name}_y1"] = segment.y1
        values[f"{name}_x2"] = segment.x2
        values[f"{name}_y2"] = segment.y2
    return values


def run_version(
    version: int,
    experiment_id: str,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    experiment_dir = args.output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    candidates_rows: list[dict[str, Any]] = []
    stage_names = [
        "hough",
        "merge",
        "connectivity",
        "angle_height",
        "ratio",
        "white_support",
        "score",
    ]
    stage_stats: dict[str, dict[str, dict[str, int]]] = {stage: {} for stage in stage_names}
    start = time.perf_counter()
    for frame_index, row in enumerate(rows, start=1):
        path = frame_path(args.frames_dir, row)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        scale = min(1.0, 960.0 / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = morphology(threshold_mask(image, config), config)
        edges, segments = make_segments(mask, config)
        stages = candidates_by_stage(mask, segments, config)
        candidates = stages[stage_names[version - 1]]
        if version == 7:
            candidates = sorted(candidates, key=lambda item: item.geometry_score, reverse=True)[:1]
        for index, stage in enumerate(stage_names):
            label_stats = stage_stats[stage].setdefault(
                row["label"],
                {
                    "frames_entering": 0,
                    "frames_surviving": 0,
                    "candidates_entering": 0,
                    "candidates_surviving": 0,
                },
            )
            entering = len(segments) if index == 0 else len(stages[stage_names[index - 1]])
            surviving = len(stages[stage])
            label_stats["frames_entering"] += int(entering > 0)
            label_stats["frames_surviving"] += int(surviving > 0)
            label_stats["candidates_entering"] += entering
            label_stats["candidates_surviving"] += surviving
        candidate = candidates[0] if candidates else None
        score = candidate.geometry_score if candidate else 0.0
        predicted = candidate is not None and (
            version < 7 or score >= config["geometry"]["min_score"]
        )
        processing_ms = (time.perf_counter() - start) * 1000 / frame_index
        frame_id = f"{row['match_id']}_{timestamp_label(float(row['timestamp_seconds']))}"
        predictions.append(
            {
                "experiment_id": experiment_id,
                "frame_id": frame_id,
                "match_id": row["match_id"],
                "true_label": row["label"],
                "goal_structure_true": row["label"] in {"main_goal", "secondary_goal"},
                "main_goal_true": row["label"] == "main_goal",
                "predicted": predicted,
                "score": score,
                "num_hough_lines": len(segments),
                "num_vertical_lines": sum(
                    abs(item.angle - 90) <= config["hough"]["vertical_tolerance"]
                    for item in segments
                ),
                "num_horizontal_lines": sum(
                    item.angle <= config["hough"]["horizontal_tolerance"] for item in segments
                ),
                "processing_ms": processing_ms,
            }
        )
        candidate_values = flatten_candidate(candidate)
        candidate_values.update(
            {"frame_id": frame_id, "candidate_id": f"{frame_id}_c001", "selected": predicted}
        )
        candidates_rows.append(candidate_values)
        if candidate is not None and predicted and version == 7:
            diagnostic = image.copy()
            for segment, color in [
                (candidate.left_post, (0, 255, 0)),
                (candidate.right_post, (0, 255, 0)),
                (candidate.crossbar, (255, 0, 0)),
            ]:
                cv2.line(
                    diagnostic,
                    (int(segment.x1), int(segment.y1)),
                    (int(segment.x2), int(segment.y2)),
                    color,
                    3,
                )
            diagnostic_dir = experiment_dir / "artifacts" / "diagnostics"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(diagnostic_dir / f"{frame_id}.jpg"), diagnostic)
    elapsed = time.perf_counter() - start
    pred_fields = list(predictions[0])
    with (experiment_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(predictions)
    candidate_fields = list(candidates_rows[0])
    with (experiment_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidates_rows)
    errors = []
    timing = []
    for prediction in predictions:
        if prediction["main_goal_true"] != prediction["predicted"]:
            errors.append(
                {
                    "frame_id": prediction["frame_id"],
                    "match_id": prediction["match_id"],
                    "error_type": "FN" if prediction["main_goal_true"] else "FP",
                    "predicted_score": prediction["score"],
                    "notes": "",
                    "manual_category": "",
                    "reviewed": "false",
                }
            )
        timing.append(
            {
                "frame_id": prediction["frame_id"],
                "processing_ms": prediction["processing_ms"],
            }
        )
    with (experiment_dir / "errors.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "frame_id",
            "match_id",
            "error_type",
            "predicted_score",
            "notes",
            "manual_category",
            "reviewed",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)
    with (experiment_dir / "timing.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame_id", "processing_ms"])
        writer.writeheader()
        writer.writerows(timing)
    with (experiment_dir / "stage_counts.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "stage",
            "label",
            "frames_entering",
            "frames_surviving",
            "candidates_entering",
            "candidates_surviving",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for stage in stage_names:
            for label, values in stage_stats[stage].items():
                writer.writerow({"stage": stage, "label": label, **values})
    metric_main = evaluate(predictions, predictions, "main_goal_true")
    metric_structure = evaluate(predictions, predictions, "goal_structure_true")
    metrics_by_match = []
    for match_id in sorted({row["match_id"] for row in predictions}):
        subset = [row for row in predictions if row["match_id"] == match_id]
        metrics_by_match.append(
            {"match_id": match_id, **evaluate(subset, subset, "main_goal_true")}
        )
    with (experiment_dir / "metrics_by_match.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(metrics_by_match[0])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics_by_match)
    metrics = {
        "experiment_id": experiment_id,
        "version": f"V{version}",
        "evaluation_main_goal": metric_main,
        "evaluation_goal_structure": metric_structure,
        "performance": {"fps": len(rows) / elapsed, "execution_seconds": elapsed},
        "frames": len(rows),
        "limitations": [
            "IoU requires bounding-box annotations.",
            "False candidates per hour requires continuous video segments.",
        ],
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (experiment_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "os": platform.platform(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "date": str(date.today()),
    }
    (experiment_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    readme = (
        f"# {experiment_id}\n\nVersao: V{version}\n\n"
        f"Dataset: {args.dataset_version}\n\n"
        "Resultado principal em `metrics.json`; candidatos instrumentados em "
        "`candidates.csv`.\n"
    )
    (experiment_dir / "README.md").write_text(readme, encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.timestamps.open(newline="", encoding="utf-8")))
    results = []
    for version in range(1, 8):
        results.append(run_version(version, f"EXP-{version + 12:04d}", rows, args, config))
    registry_header = [
        "experiment_id",
        "date",
        "stage",
        "version_name",
        "git_commit",
        "dataset_version",
        "config_file",
        "hardware",
        "status",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_positive_rate",
        "fps",
        "execution_seconds",
        "notes",
    ]
    registry_rows = []
    for result in results:
        metrics = result["evaluation_main_goal"]
        registry_rows.append(
            {
                "experiment_id": result["experiment_id"],
                "date": str(date.today()),
                "stage": "goal_geometry",
                "version_name": result["version"],
                "git_commit": "working-tree",
                "dataset_version": args.dataset_version,
                "config_file": f"{result['experiment_id']}/config.yaml",
                "hardware": "CPU",
                "status": "completed",
                "precision": f"{metrics['precision']:.4f}",
                "recall": f"{metrics['recall']:.4f}",
                "f1": f"{metrics['f1']:.4f}",
                "false_positives": metrics["fp"],
                "false_positive_rate": (
                    f"{metrics['fp'] / (metrics['fp'] + metrics['tn']):.4f}"
                    if metrics["fp"] + metrics["tn"]
                    else ""
                ),
                "fps": f"{result['performance']['fps']:.2f}",
                "execution_seconds": f"{result['performance']['execution_seconds']:.3f}",
                "notes": "Comparacao incremental V1-V7; metricas de FP/h e IoU pendentes.",
            }
        )
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        list(csv.DictReader(args.registry.open(newline="", encoding="utf-8")))
        if args.registry.exists()
        else []
    )
    by_id = {row["experiment_id"]: row for row in existing}
    by_id.update({row["experiment_id"]: row for row in registry_rows})
    with args.registry.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=registry_header)
        writer.writeheader()
        writer.writerows(by_id.values())
    print(
        json.dumps(
            {"experiments": [result["experiment_id"] for result in results]}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
