"""Confirma uma única vez em VALIDATION o ranker temporal enriquecido treinado em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    parser = argparse.ArgumentParser(description="Confirma ranker enriquecido uma vez em VALIDATION.")
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    train_rows, train_fields = read_csv(args.train_dataset)
    validation_rows, validation_fields = read_csv(args.validation_dataset)
    excluded = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    feature_names = [field for field in train_fields if field not in excluded]
    validation_features = [field for field in validation_fields if field not in excluded]
    if feature_names != validation_features:
        raise ValueError("TRAIN e VALIDATION não possuem o mesmo contrato de features.")
    catalog, _ = read_csv(args.validation_goal_timestamps)
    if any(row["split"] != "validation" for row in catalog):
        raise AssertionError("A confirmação aceita exclusivamente VALIDATION; TEST permanece fechado.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    x_train = np.asarray([[float(row[name]) for name in feature_names] for row in train_rows], dtype=np.float32)
    y_train = np.asarray([int(row["weak_label"]) for row in train_rows], dtype=int)
    x_validation = np.asarray([[float(row[name]) for name in feature_names] for row in validation_rows], dtype=np.float32)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_validation)[:, 1]
    predicted = [{**row, "rank_score": f"{float(score):.8f}"} for row, score in zip(validation_rows, scores)]
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predicted:
        by_match[str(row["match_id"])].append(row)
    limits = sorted(set(args.candidate_limits))
    goal_ranks: list[dict[str, object]] = []
    for match_id, goals in sorted(goals_by_match.items()):
        chosen = select_diverse(by_match[match_id], max(limits), args.min_separation_seconds)
        for goal in goals:
            ranks = [index for index, row in enumerate(chosen, start=1) if float(row["best_timestamp_seconds"]) - args.clip_pre_seconds <= goal <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds]
            goal_ranks.append({"match_id": match_id, "goal_timestamp_seconds": f"{goal:.3f}", "best_diverse_candidate_rank": ranks[0] if ranks else ""})
    recall_rows: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    for limit in limits:
        covered = sum(bool(row["best_diverse_candidate_rank"]) and int(row["best_diverse_candidate_rank"]) <= limit for row in goal_ranks)
        recall_rows.append({"k": limit, "catalog_goals": len(goal_ranks), "covered_goals": covered, "recall_at_k": f"{covered / len(goal_ranks):.6f}"})
        for match_id in sorted(goals_by_match):
            ranks = [row["best_diverse_candidate_rank"] for row in goal_ranks if row["match_id"] == match_id]
            match_covered = sum(bool(rank) and int(rank) <= limit for rank in ranks)
            per_match_rows.append({"k": limit, "match_id": match_id, "catalog_goals": len(ranks), "covered_goals": match_covered, "recall_at_k": f"{match_covered / len(ranks):.6f}"})
    summary = {
        "experiment_id": args.output.name,
        "scope": "modelo treinado somente em TRAIN; confirmação única em VALIDATION",
        "model": "RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight=balanced, random_state=42)",
        "feature_count": len(feature_names),
        "trajectory_feature_count": sum(name.startswith("trajectory_") for name in feature_names),
        "warning": "Este resultado é confirmação única, não autorização para novo tuning; TEST não foi acessado.",
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "validation_predictions_v1.csv", predicted, [*validation_fields, "rank_score"])
    write_csv(args.output / "recall_at_k_validation_v1.csv", recall_rows, ["k", "catalog_goals", "covered_goals", "recall_at_k"])
    write_csv(args.output / "per_match_recall_validation_v1.csv", per_match_rows, ["k", "match_id", "catalog_goals", "covered_goals", "recall_at_k"])
    write_csv(args.output / "goal_ranks_validation_v1.csv", goal_ranks, ["match_id", "goal_timestamp_seconds", "best_diverse_candidate_rank"])
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recall_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
