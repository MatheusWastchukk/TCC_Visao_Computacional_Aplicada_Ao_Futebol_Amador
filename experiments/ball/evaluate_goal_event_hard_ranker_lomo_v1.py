"""Avalia o ranker com negativos difíceis usando LOMO, somente em TRAIN."""
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
    parser = argparse.ArgumentParser(description="Avalia Random Forest com negativos difíceis em LOMO TRAIN.")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    limits = sorted(set(args.candidate_limits))
    rows, fields = read_csv(args.selection)
    required = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label", "selected_for_hard_ranker"}
    if missing := required - set(fields):
        raise ValueError(f"Seleção sem colunas: {sorted(missing)}")
    feature_names = [field for field in fields if field not in {"group_uid", "match_id", "best_timestamp_seconds", "weak_label", "base_lomo_rank_score", "contains_catalog_goal_in_clip", "selection_role", "selected_for_hard_ranker"}]
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("A avaliação do hard ranker aceita somente TRAIN.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["weak_label"]) for row in rows], dtype=int)
    matches = np.asarray([row["match_id"] for row in rows])
    selected = np.asarray([row["selected_for_hard_ranker"] == "True" for row in rows])
    scores = np.zeros(len(rows), dtype=np.float64)
    for held_match in sorted(set(matches)):
        train_mask, test_mask = (matches != held_match) & selected, matches == held_match
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(x[train_mask], y[train_mask])
        scores[test_mask] = model.predict_proba(x[test_mask])[:, 1]
    predicted = [{**row, "rank_score": f"{float(score):.8f}"} for row, score in zip(rows, scores)]
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predicted:
        by_match[str(row["match_id"])].append(row)
    goal_ranks: list[dict[str, object]] = []
    for match_id, goals in goals_by_match.items():
        selected_rows = select_diverse(by_match[match_id], max(limits), args.min_separation_seconds)
        for goal in goals:
            ranks = [index for index, row in enumerate(selected_rows, start=1) if float(row["best_timestamp_seconds"]) - args.clip_pre_seconds <= goal <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds]
            goal_ranks.append({"match_id": match_id, "goal_timestamp_seconds": f"{goal:.3f}", "best_diverse_candidate_rank": ranks[0] if ranks else ""})
    recall_rows: list[dict[str, object]] = []
    for limit in limits:
        covered = sum(int(row["best_diverse_candidate_rank"] != "" and int(row["best_diverse_candidate_rank"]) <= limit) for row in goal_ranks)
        recall_rows.append({"k": limit, "catalog_goals": len(goal_ranks), "covered_goals": covered, "recall_at_k": f"{covered / len(goal_ranks):.6f}"})
    args.output.mkdir(parents=True)
    write_csv(args.output / "lomo_predictions_v1.csv", predicted, [*fields, "rank_score"])
    write_csv(args.output / "recall_at_k_lomo_v1.csv", recall_rows, ["k", "catalog_goals", "covered_goals", "recall_at_k"])
    write_csv(args.output / "goal_ranks_lomo_v1.csv", goal_ranks, list(goal_ranks[0]))
    summary = {"experiment_id": args.output.name, "scope": "somente TRAIN; leave-one-match-out", "features": feature_names, "selected_training_rows": int(selected.sum()), "warning": "Hipótese A: somente a composição de negativos muda; nenhum dado VALIDATION ou TEST foi utilizado."}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recall_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
