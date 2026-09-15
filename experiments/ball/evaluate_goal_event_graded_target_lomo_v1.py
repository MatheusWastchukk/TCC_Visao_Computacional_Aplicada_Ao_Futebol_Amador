"""Avalia alvo temporal graduado com LOMO, somente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relevance_from_distance(distance_seconds: float) -> float:
    """Converte a distância temporal na escala fixada antes do experimento."""
    if distance_seconds <= 2.0:
        return 1.0
    if distance_seconds <= 5.0:
        return 0.75
    if distance_seconds <= 10.0:
        return 0.40
    return 0.0


def select_diverse(rows: list[dict[str, object]], limit: int, separation: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia alvo temporal graduado em LOMO TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    rows, fields = read_csv(args.dataset)
    required = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    if missing := required - set(fields):
        raise ValueError(f"Dataset sem colunas: {sorted(missing)}")
    catalog, _ = read_csv(args.goal_timestamps)
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("A hipótese B aceita somente timestamps TRAIN.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if set(row["match_id"] for row in rows) != set(goals_by_match):
        raise ValueError("Partidas do dataset e do catálogo TRAIN não coincidem.")

    excluded = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    feature_names = [field for field in fields if field not in excluded]
    distances = np.asarray([
        min(abs(float(row["best_timestamp_seconds"]) - goal) for goal in goals_by_match[row["match_id"]])
        for row in rows
    ], dtype=np.float32)
    targets = np.asarray([relevance_from_distance(float(distance)) for distance in distances], dtype=np.float32)
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    matches = np.asarray([row["match_id"] for row in rows])
    scores = np.zeros(len(rows), dtype=np.float64)
    for held_match in sorted(set(matches)):
        train_mask, test_mask = matches != held_match, matches == held_match
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(x[train_mask], targets[train_mask])
        scores[test_mask] = model.predict(x[test_mask])

    predicted = [
        {
            **row,
            "nearest_catalog_goal_distance_seconds": f"{float(distance):.6f}",
            "graded_relevance_target": f"{float(target):.2f}",
            "rank_score": f"{float(score):.8f}",
        }
        for row, distance, target, score in zip(rows, distances, targets, scores)
    ]
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predicted:
        by_match[str(row["match_id"])].append(row)
    limits = sorted(set(args.candidate_limits))
    goal_ranks: list[dict[str, object]] = []
    for match_id, goals in sorted(goals_by_match.items()):
        selected_rows = select_diverse(by_match[match_id], max(limits), args.min_separation_seconds)
        for goal in goals:
            ranks = [index for index, row in enumerate(selected_rows, start=1) if float(row["best_timestamp_seconds"]) - args.clip_pre_seconds <= goal <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds]
            goal_ranks.append({"match_id": match_id, "goal_timestamp_seconds": f"{goal:.3f}", "best_diverse_candidate_rank": ranks[0] if ranks else ""})
    recall_rows: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    for limit in limits:
        covered = sum(bool(row["best_diverse_candidate_rank"]) and int(row["best_diverse_candidate_rank"]) <= limit for row in goal_ranks)
        recall_rows.append({"k": limit, "catalog_goals": len(goal_ranks), "covered_goals": covered, "recall_at_k": f"{covered / len(goal_ranks):.6f}"})
        for match_id in sorted(goals_by_match):
            match_ranks = [row["best_diverse_candidate_rank"] for row in goal_ranks if row["match_id"] == match_id]
            match_covered = sum(bool(rank) and int(rank) <= limit for rank in match_ranks)
            per_match_rows.append({"k": limit, "match_id": match_id, "catalog_goals": len(match_ranks), "covered_goals": match_covered, "recall_at_k": f"{match_covered / len(match_ranks):.6f}"})
    numeric_ranks = [int(row["best_diverse_candidate_rank"]) for row in goal_ranks if row["best_diverse_candidate_rank"]]
    summary = {
        "experiment_id": args.output.name,
        "scope": "somente TRAIN; leave-one-match-out",
        "model": "RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)",
        "target_definition": {"0-2_seconds": 1.0, "2-5_seconds": 0.75, "5-10_seconds": 0.40, "over_10_seconds": 0.0},
        "feature_count": len(feature_names),
        "target_counts": {str(value): int((targets == value).sum()) for value in sorted(set(targets))},
        "median_rank_when_covered": float(np.median(numeric_ranks)) if numeric_ranks else None,
        "warning": "Hipótese B: somente o target de treino muda; nenhum dado VALIDATION ou TEST foi utilizado.",
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "lomo_predictions_v1.csv", predicted, [*fields, "nearest_catalog_goal_distance_seconds", "graded_relevance_target", "rank_score"])
    write_csv(args.output / "recall_at_k_lomo_v1.csv", recall_rows, ["k", "catalog_goals", "covered_goals", "recall_at_k"])
    write_csv(args.output / "per_match_recall_lomo_v1.csv", per_match_rows, ["k", "match_id", "catalog_goals", "covered_goals", "recall_at_k"])
    write_csv(args.output / "goal_ranks_lomo_v1.csv", goal_ranks, ["match_id", "goal_timestamp_seconds", "best_diverse_candidate_rank"])
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recall_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
