"""Deriva atributos de trajetória dos quatro bins temporais já congelados."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BASE_COLUMNS = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def main() -> int:
    parser = argparse.ArgumentParser(description="Acrescenta trajetória temporal ao dataset TRAIN existente.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["train", "validation"], default="train")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    rows, fields = read_csv(args.dataset)
    required = {f"bin_{index}_{name}" for index in range(4) for name in ("mean_distance", "mean_score", "mean_ball_confidence", "inside_fraction")}
    if missing := required - set(fields):
        raise ValueError(f"Dataset não contém os quatro bins esperados: {sorted(missing)}")
    derived_names = [
        "trajectory_distance_delta_0_3", "trajectory_score_delta_0_3",
        "trajectory_ball_confidence_delta_0_3", "trajectory_inside_delta_0_3",
        "trajectory_distance_abs_change_sum", "trajectory_score_abs_change_sum",
        "trajectory_distance_decrease_fraction", "trajectory_score_increase_fraction",
        "trajectory_distance_slope", "trajectory_score_slope", "trajectory_ball_confidence_slope",
        "trajectory_distance_min_bin", "trajectory_score_max_bin",
    ]
    enriched: list[dict[str, object]] = []
    for row in rows:
        distance = np.asarray([float(row[f"bin_{index}_mean_distance"]) for index in range(4)], dtype=np.float64)
        score = np.asarray([float(row[f"bin_{index}_mean_score"]) for index in range(4)], dtype=np.float64)
        ball = np.asarray([float(row[f"bin_{index}_mean_ball_confidence"]) for index in range(4)], dtype=np.float64)
        inside = np.asarray([float(row[f"bin_{index}_inside_fraction"]) for index in range(4)], dtype=np.float64)
        values = {
            "trajectory_distance_delta_0_3": distance[3] - distance[0],
            "trajectory_score_delta_0_3": score[3] - score[0],
            "trajectory_ball_confidence_delta_0_3": ball[3] - ball[0],
            "trajectory_inside_delta_0_3": inside[3] - inside[0],
            "trajectory_distance_abs_change_sum": float(np.abs(np.diff(distance)).sum()),
            "trajectory_score_abs_change_sum": float(np.abs(np.diff(score)).sum()),
            "trajectory_distance_decrease_fraction": float(np.mean(np.diff(distance) <= 0.0)),
            "trajectory_score_increase_fraction": float(np.mean(np.diff(score) >= 0.0)),
            "trajectory_distance_slope": float(np.polyfit(np.arange(4), distance, 1)[0]),
            "trajectory_score_slope": float(np.polyfit(np.arange(4), score, 1)[0]),
            "trajectory_ball_confidence_slope": float(np.polyfit(np.arange(4), ball, 1)[0]),
            "trajectory_distance_min_bin": float(np.argmin(distance) / 3.0),
            "trajectory_score_max_bin": float(np.argmax(score) / 3.0),
        }
        enriched.append({**row, **{name: f"{value:.6f}" for name, value in values.items()}})
    args.output.mkdir(parents=True)
    with (args.output / "temporal_group_dataset_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*fields, *derived_names])
        writer.writeheader()
        writer.writerows(enriched)
    summary = {
        "experiment_id": args.output.name,
        "scope": f"somente {args.expected_split.upper()}; derivado sem novos dados",
        "groups": len(enriched),
        "original_feature_count": len(fields) - len(BASE_COLUMNS),
        "derived_trajectory_features": derived_names,
        "note": "A fonte histórica conserva quatro bins normalizados; a hipótese C testa explicitamente tendências, variações e posições de pico nesses bins, mantendo todos os 5.742 grupos comparáveis.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
