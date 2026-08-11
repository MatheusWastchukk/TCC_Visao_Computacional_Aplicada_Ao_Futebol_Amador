"""Executa a V8 usando pontuações geométricas contínuas calibradas com rótulos de quatro pontos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from experiments.goal_geometry.run_geometry_versions import (
    candidate_pool,
    evaluate,
    frame_path,
    make_segments,
    morphology,
    threshold_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa a pontuação de geometria suave da V8.")
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def annotation_features(row: dict[str, str]) -> dict[str, float]:
    lt = (float(row["lt_x"]), float(row["lt_y"]))
    rt = (float(row["rt_x"]), float(row["rt_y"]))
    lb = (float(row["lb_x"]), float(row["lb_y"]))
    rb = (float(row["rb_x"]), float(row["rb_y"]))
    left_height = distance(lt, lb)
    right_height = distance(rt, rb)
    return {
        "ratio": distance(lt, rt) / max((left_height + right_height) / 2, 1.0),
        "relative_width": distance(lt, rt) / 1920.0,
        "relative_height": ((left_height + right_height) / 2) / 1080.0,
    }


def calibration(annotations: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    values = {
        key: [annotation_features(row)[key] for row in annotations]
        for key in ["ratio", "relative_width", "relative_height"]
    }
    result: dict[str, dict[str, float]] = {}
    for key, numbers in values.items():
        result[key] = {
            "min": min(numbers),
            "max": max(numbers),
            "mean": float(np.mean(numbers)),
            "median": float(np.median(numbers)),
            "std": max(float(np.std(numbers)), 0.05 * float(np.median(numbers))),
            "p05": float(np.percentile(numbers, 5)),
            "p95": float(np.percentile(numbers, 95)),
        }
    return result


def gaussian_score(value: float, center: float, scale: float) -> float:
    return math.exp(-0.5 * ((value - center) / max(scale, 1e-6)) ** 2)


def soft_score(candidate: Any, stats: dict[str, dict[str, float]]) -> dict[str, float]:
    ratio_score = gaussian_score(
        candidate.width_height_ratio, stats["ratio"]["median"], stats["ratio"]["std"]
    )
    size_score = gaussian_score(
        candidate.relative_height,
        stats["relative_height"]["median"],
        stats["relative_height"]["std"],
    )
    scores = {
        "connection_score": candidate.connection_score,
        "angle_score": candidate.angle_score,
        "height_score": candidate.height_score,
        "ratio_score": ratio_score,
        "white_support_score": candidate.white_support_score,
        "size_score": size_score,
    }
    scores["geometry_score"] = sum(
        weight * scores[key]
        for key, weight in {
            "connection_score": 0.20,
            "angle_score": 0.15,
            "height_score": 0.15,
            "ratio_score": 0.20,
            "white_support_score": 0.15,
            "size_score": 0.15,
        }.items()
    )
    return scores


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.timestamps.open(newline="", encoding="utf-8")))
    annotations = list(csv.DictReader(args.annotations.open(newline="", encoding="utf-8")))
    if len(annotations) != 35 or any(
        row["annotation_status"] != "completed" for row in annotations
    ):
        raise ValueError("A V8 exige 35 anotações completas com annotation_status=completed")
    stats = calibration(annotations)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    thresholds = [round(value, 2) for value in np.arange(0.30, 0.86, 0.05)]
    all_candidates: list[dict[str, Any]] = []
    start = time.perf_counter()
    for row in rows:
        image = cv2.imread(str(frame_path(args.frames_dir, row)))
        if image is None:
            raise FileNotFoundError(frame_path(args.frames_dir, row))
        scale = min(1.0, 960.0 / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = morphology(threshold_mask(image, config), config)
        _edges, segments = make_segments(mask, config)
        candidates = candidate_pool(mask, segments, config, merged=True)
        frame_id = f"{row['match_id']}_{float(row['timestamp_seconds']):010.3f}"
        for index, candidate in enumerate(candidates):
            scores = soft_score(candidate, stats)
            all_candidates.append(
                {
                    "frame_id": frame_id,
                    "match_id": row["match_id"],
                    "true_label": row["label"],
                    "candidate_id": f"{frame_id}_c{index + 1:03d}",
                    **scores,
                    "processing_ms": 0.0,
                }
            )
    elapsed = time.perf_counter() - start
    for candidate_row in all_candidates:
        candidate_row["processing_ms"] = elapsed * 1000 / max(len(rows), 1)
    predictions_by_threshold: dict[str, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        predictions = []
        for row in rows:
            frame_id = f"{row['match_id']}_{float(row['timestamp_seconds']):010.3f}"
            frame_candidates = [item for item in all_candidates if item["frame_id"] == frame_id]
            best = max(frame_candidates, key=lambda item: item["geometry_score"], default=None)
            predictions.append(
                {
                    "frame_id": frame_id,
                    "match_id": row["match_id"],
                    "true_label": row["label"],
                    "main_goal_true": row["label"] == "main_goal",
                    "goal_structure_true": row["label"] in {"main_goal", "secondary_goal"},
                    "predicted": bool(best and best["geometry_score"] >= threshold),
                    "score": best["geometry_score"] if best else 0.0,
                }
            )
        predictions_by_threshold[f"{threshold:.2f}"] = predictions
    sweep = []
    for threshold, predictions in predictions_by_threshold.items():
        main_metrics = evaluate(predictions, predictions, "main_goal_true")
        structure_metrics = evaluate(predictions, predictions, "goal_structure_true")
        sweep.append(
            {
                "threshold": threshold,
                **{f"main_{key}": value for key, value in main_metrics.items()},
                **{f"structure_{key}": value for key, value in structure_metrics.items()},
            }
        )
    best = max(sweep, key=lambda row: row["structure_f1"])
    best_threshold = str(best["threshold"])
    best_predictions = predictions_by_threshold[best_threshold]
    with (output / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(best_predictions[0]))
        writer.writeheader()
        writer.writerows(best_predictions)
    by_match = []
    for match_id in sorted({row["match_id"] for row in best_predictions}):
        subset = [row for row in best_predictions if row["match_id"] == match_id]
        by_match.append({"match_id": match_id, **evaluate(subset, subset, "goal_structure_true")})
    with (output / "metrics_by_match.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(by_match[0]))
        writer.writeheader()
        writer.writerows(by_match)
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_candidates[0]))
        writer.writeheader()
        writer.writerows(all_candidates)
    with (output / "timing.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame_id", "processing_ms"])
        writer.writeheader()
        writer.writerows(
            {"frame_id": row["frame_id"], "processing_ms": len(rows) / elapsed}
            for row in best_predictions
        )
    with (output / "errors.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["frame_id", "match_id", "error_type", "predicted_score", "reviewed"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                "frame_id": row["frame_id"],
                "match_id": row["match_id"],
                "error_type": "FN" if row["goal_structure_true"] else "FP",
                "predicted_score": row["score"],
                "reviewed": "false",
            }
            for row in best_predictions
            if row["goal_structure_true"] != row["predicted"]
        )
    report = {
        "experiment_id": "EXP-0020",
        "version": "V8",
        "method": "soft_geometry_scoring",
        "frames": len(rows),
        "candidates": len(all_candidates),
        "best_threshold": best,
        "performance": {"fps": len(rows) / elapsed, "execution_seconds": elapsed},
        "weights": {
            "connection": 0.20,
            "angle": 0.15,
            "height": 0.15,
            "ratio": 0.20,
            "white_support": 0.15,
            "size": 0.15,
        },
        "limitations": [
            "IoU requires bounding-box annotations; four-point labels are used for calibration."
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "os": platform.platform(),
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip(),
                "date": str(date.today()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# EXP-0020 - V8 Soft Geometry\n\n"
            "Score contínuo calibrado com as 35 anotações de quatro pontos. "
        "O threshold escolhido maximiza F1 de goal_structure no sweep.\n",
        encoding="utf-8",
    )
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
    registry_rows = (
        list(csv.DictReader(args.registry.open(newline="", encoding="utf-8")))
        if args.registry.exists()
        else []
    )
    registry_rows = [row for row in registry_rows if row["experiment_id"] != "EXP-0020"]
    best_fp = int(best["structure_fp"])
    best_tn = int(best["structure_tn"])
    registry_rows.append(
        {
            "experiment_id": "EXP-0020",
            "date": str(date.today()),
            "stage": "goal_geometry",
            "version_name": "V8_soft_geometry",
            "git_commit": "working-tree",
            "dataset_version": "dataset_frames_v1",
            "config_file": "EXP-0020/config.yaml",
            "hardware": "CPU",
            "status": "completed",
            "precision": f"{best['structure_precision']:.4f}",
            "recall": f"{best['structure_recall']:.4f}",
            "f1": f"{best['structure_f1']:.4f}",
            "false_positives": best_fp,
            "false_positive_rate": f"{best_fp / (best_fp + best_tn):.4f}",
            "fps": f"{len(rows) / elapsed:.2f}",
            "execution_seconds": f"{elapsed:.3f}",
            "notes": "Avaliacao principal goal_structure; threshold calibrado no sweep.",
        }
    )
    with args.registry.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=registry_header)
        writer.writeheader()
        writer.writerows(registry_rows)
    print(json.dumps({"experiment_id": "EXP-0020", "best_threshold": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
