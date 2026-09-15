"""Aplica a fusão congelada a candidatos já pontuados visualmente em VALIDATION."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aplica fusão congelada sem recalibração em VALIDATION.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["train", "validation"], required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    limits = sorted(set(args.candidate_limits))
    candidates, fields = read_csv(args.candidate_scores)
    required = {"event_id", "match_id", "split", "candidate_start_seconds", "candidate_end_seconds", "rank_score", "visual_score_goal"}
    if missing := required - set(fields):
        raise ValueError(f"Candidatos sem colunas: {sorted(missing)}")
    if not candidates or any(row["split"] != args.expected_split for row in candidates):
        raise AssertionError("A fusão exige candidatos de um único split esperado.")
    frozen = joblib.load(args.model)
    if frozen.get("features") != ["rank_score", "visual_score_goal"]:
        raise ValueError("Contrato incompatível do modelo de fusão.")
    x = np.asarray([[float(row["rank_score"]), float(row["visual_score_goal"])] for row in candidates], dtype=np.float64)
    scores = frozen["model"].predict_proba(x)[:, 1]
    scored = [{**row, "fusion_score_goal": f"{float(score):.8f}"} for row, score in zip(candidates, scores)]

    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"goal_catalog_id", "match_id", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Catálogo sem colunas: {sorted(missing)}")
    if any(row["split"] != args.expected_split for row in catalog):
        raise AssertionError("O catálogo contém split incompatível.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    all_goals = sum(len(values) for values in goals_by_match.values())
    comparison: list[dict[str, object]] = []
    for method in ("rank_score", "visual_score_goal", "fusion_score_goal"):
        for limit in limits:
            covered = 0
            for match_id, goals in goals_by_match.items():
                selected = sorted([row for row in scored if row["match_id"] == match_id], key=lambda item: float(item[method]), reverse=True)[:limit]
                covered += sum(int(any(float(row["candidate_start_seconds"]) <= goal <= float(row["candidate_end_seconds"]) for row in selected)) for goal in goals)
            comparison.append({"method": method, "k": limit, "catalog_goals": all_goals, "covered_goals": covered, "recall_at_k": f"{covered / all_goals:.6f}"})

    args.output.mkdir(parents=True)
    write_csv(args.output / "candidate_fusion_scores_v1.csv", scored, [*fields, "fusion_score_goal"])
    write_csv(args.output / "recall_at_k_comparison_v1.csv", comparison, ["method", "k", "catalog_goals", "covered_goals", "recall_at_k"])
    summary = {"experiment_id": args.output.name, "scope": "VALIDATION; configuração TRAIN congelada", "candidates": len(scored), "candidate_limits": limits, "warning": "Confirmação única de fusão congelada; não ajustar limiares ou pesos com estes resultados."}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
