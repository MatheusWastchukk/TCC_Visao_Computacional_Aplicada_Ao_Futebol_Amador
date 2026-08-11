"""Executa o baseline clássico da estrutura da goleira nos frames do piloto.

O baseline é intencionalmente explicável: limiarização de pixels claros,
morfologia opcional, bordas de Canny e geometria de linhas de Hough. Ele
classifica os rótulos ``main_goal``, ``not_goal`` e ``secondary_goal``; não
afirma provar que a bola cruzou a linha oficial do gol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Detection:
    positive: bool
    confidence: float
    vertical_lines: int
    horizontal_lines: int


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    secondary_rejection_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o baseline clássico da goleira.")
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def timestamp_label(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    whole_seconds, ms = divmod(milliseconds, 1000)
    return f"{whole_seconds:06d}.{ms:03d}"


def frame_path(frames_dir: Path, row: dict[str, str]) -> Path:
    label = timestamp_label(float(row["timestamp_seconds"]))
    return frames_dir / row["label"] / f"{row['match_id']}_{label}.jpg"


def threshold_mask(image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    color_space = variant["color_space"]
    method = variant["method"]
    if color_space == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        hsv_mask = ((value >= int(variant.get("min_value", 190))) &
                    (saturation <= int(variant.get("max_saturation", 70))))
        return hsv_mask.astype(np.uint8) * 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    _, gray_mask = cv2.threshold(
        gray, int(variant.get("min_value", 190)), 255, cv2.THRESH_BINARY
    )
    return gray_mask


def apply_morphology(mask: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    operation = variant["operation"]
    kernel_size = int(variant.get("kernel", 0))
    if operation == "none" or kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if operation == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if operation == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if operation == "open_close":
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    raise ValueError(f"Operação morfológica desconhecida: {operation}")


def _line_features(mask: np.ndarray, config: dict[str, Any]) -> tuple[int, int, float]:
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=int(config["min_line_length"]),
        maxLineGap=int(config["max_line_gap"]),
    )
    if lines is None:
        return 0, 0, 0.0

    vertical = 0
    horizontal = 0
    vertical_lengths: list[float] = []
    horizontal_lengths: list[float] = []
    vertical_tolerance = float(config["vertical_angle_tolerance"])
    horizontal_tolerance = float(config["horizontal_angle_tolerance"])
    for item in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, item)
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180.0 - angle)
        length = math.hypot(x2 - x1, y2 - y1)
        if abs(angle - 90.0) <= vertical_tolerance:
            vertical += 1
            vertical_lengths.append(length)
        elif angle <= horizontal_tolerance:
            horizontal += 1
            horizontal_lengths.append(length)

    # Duas traves verticais e uma trave horizontal formam a estrutura mínima interpretável.
    post_score = min(1.0, vertical / 2.0)
    bar_score = min(1.0, horizontal / 1.0)
    length_score = min(1.0, (sum(vertical_lengths) + sum(horizontal_lengths)) / 500.0)
    confidence = 0.45 * post_score + 0.35 * bar_score + 0.20 * length_score
    return vertical, horizontal, confidence


def detect(
    image: np.ndarray,
    threshold: dict[str, Any],
    morphology: dict[str, Any],
    geometry: dict[str, Any],
) -> Detection:
    scale = min(1.0, 960.0 / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = threshold_mask(image, threshold)
    mask = apply_morphology(mask, morphology)
    vertical, horizontal, confidence = _line_features(mask, geometry)
    return Detection(
        positive=confidence >= float(geometry.get("min_confidence", 0.65)),
        confidence=confidence,
        vertical_lines=vertical,
        horizontal_lines=horizontal,
    )


def calculate_metrics(rows: list[dict[str, Any]]) -> Metrics:
    tp = sum(row["expected"] and row["predicted"] for row in rows)
    tn = sum(not row["expected"] and not row["predicted"] for row in rows)
    fp = sum(not row["expected"] and row["predicted"] for row in rows)
    fn = sum(row["expected"] and not row["predicted"] for row in rows)
    secondary = [row for row in rows if row["label"] == "secondary_goal"]
    secondary_rejected = sum(not row["predicted"] for row in secondary)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fp / (fp + tn) if fp + tn else 0.0,
        secondary_rejection_rate=secondary_rejected / len(secondary) if secondary else 1.0,
    )


def _variants(config: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (threshold, morphology)
        for threshold in config["threshold"]["variants"]
        for morphology in config["morphology"]["variants"]
    ]


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.timestamps.open(newline="", encoding="utf-8")))
    matrix_rows: list[dict[str, Any]] = []
    report_variants: list[dict[str, Any]] = []

    for threshold, morphology in _variants(config):
        start = time.perf_counter()
        evaluated: list[dict[str, Any]] = []
        for row in rows:
            path = frame_path(args.frames_dir, row)
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Frame não encontrado ou ilegível: {path}")
            result = detect(image, threshold, morphology, config["lines"] | config["geometry"])
            evaluated.append({
                "match_id": row["match_id"],
                "label": row["label"],
                "expected": row["label"] == "main_goal",
                "predicted": result.positive,
            })
        elapsed = time.perf_counter() - start
        metrics = calculate_metrics(evaluated)
        experiment_id = f"{threshold['id']}__{morphology['id']}"
        per_match: dict[str, Metrics] = {
            match_id: calculate_metrics([row for row in evaluated if row["match_id"] == match_id])
            for match_id in sorted({row["match_id"] for row in evaluated})
        }
        for match_id, match_metrics in per_match.items():
            matrix_rows.append({
                "experiment_id": experiment_id,
                "color_space": threshold["color_space"],
                "threshold_method": threshold["method"],
                "threshold_values": json.dumps(threshold, ensure_ascii=False, sort_keys=True),
                "morphology": morphology["operation"],
                "kernel": morphology.get("kernel", 0),
                "edge_parameters": "canny_50_150",
                "hough_parameters": "probabilistic_hough",
                "geometry_parameters": "two_posts_plus_crossbar",
                "confidence_threshold": config["geometry"]["min_confidence"],
                "match_id": match_id,
                "precision": f"{match_metrics.precision:.4f}",
                "recall": f"{match_metrics.recall:.4f}",
                "f1": f"{match_metrics.f1:.4f}",
                "mean_iou": "",
                "false_candidates_per_hour": "",
                "processing_fps": f"{len(rows) / elapsed:.2f}",
                "notes": (
                    "IoU/FP-hour dependem de anotações temporais e caixas; "
                    "o baseline avalia a classificação de frames."
                ),
            })
        report_variants.append({
            "experiment_id": experiment_id,
            "threshold": threshold,
            "morphology": morphology,
            "metrics": metrics.__dict__,
            "elapsed_seconds": round(elapsed, 3),
            "processing_fps": round(len(rows) / elapsed, 2),
        })

    fieldnames = list(matrix_rows[0])
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    with args.matrix_output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)
    best = max(report_variants, key=lambda item: item["metrics"]["f1"])
    report = {
        "baseline": "classical_threshold_morphology_hough",
        "target": "main_goal_vs_not_goal_and_secondary_goal",
        "frames_evaluated": len(rows),
        "variants_evaluated": len(report_variants),
        "best_variant": best,
        "variants": report_variants,
        "limitations": [
            "Não havia anotações de caixas delimitadoras; portanto, o IoU médio não foi calculado.",
            (
                "A amostra de timestamps não é um segmento contínuo de vídeo; portanto, os "
                "falsos candidatos por hora não foram calculados."
            ),
            "Os rótulos são referências no nível de frame e não comprovam a transposição da linha oficial do gol.",
        ],
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"best_variant": best, "frames": len(rows), "variants": len(report_variants)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
