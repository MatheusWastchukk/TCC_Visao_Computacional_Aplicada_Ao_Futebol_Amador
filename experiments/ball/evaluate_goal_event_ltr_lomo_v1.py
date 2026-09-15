"""Avalia um único Learning-to-Rank por partida, exclusivamente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from xgboost import XGBRanker


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
    parser = argparse.ArgumentParser(description="Avalia XGBoost rank:pairwise em leave-one-match-out, somente TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20, 30])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    limits = sorted(set(args.candidate_limits))
    if not limits or limits[0] < 1:
        raise ValueError("candidate-limits deve conter inteiros positivos.")

    rows, fields = read_csv(args.dataset)
    required = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    if missing := required - set(fields):
        raise ValueError(f"Dataset sem colunas: {sorted(missing)}")
    feature_names = [field for field in fields if field not in required]
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"goal_catalog_id", "match_id", "split", "event_timestamp_seconds", "annotation_status"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Catálogo sem colunas: {sorted(missing)}")
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("O experimento LTR aceita somente timestamps TRAIN; VALIDATION e TEST permanecem fechados.")
    goals = [row for row in catalog if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]]
    if not goals:
        raise ValueError("Não há gols TRAIN completed para avaliar.")

    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["weak_label"]) for row in rows], dtype=np.int32)
    matches = np.asarray([row["match_id"] for row in rows])
    predictions = np.zeros(len(rows), dtype=np.float64)
    configuration = {"objective": "rank:pairwise", "n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1.0, "reg_lambda": 1.0, "tree_method": "hist", "device": "cpu", "random_state": 42, "n_jobs": 4}
    for held_match in sorted(set(matches)):
        train_mask, test_mask = matches != held_match, matches == held_match
        train_indices = np.where(train_mask)[0]
        ordered_train = train_indices[np.argsort(matches[train_indices], kind="stable")]
        ordered_matches = matches[ordered_train]
        group_sizes = [int(np.sum(ordered_matches == match)) for match in sorted(set(ordered_matches))]
        ranker = XGBRanker(**configuration)
        ranker.fit(x[ordered_train], y[ordered_train], group=group_sizes, verbose=False)
        predictions[test_mask] = ranker.predict(x[test_mask])

    predicted_rows = [{**row, "rank_score": f"{float(score):.8f}"} for row, score in zip(rows, predictions)]
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predicted_rows:
        by_match[str(row["match_id"])].append(row)
    selected_by_match = {match_id: select_diverse(match_rows, max(limits), args.min_separation_seconds) for match_id, match_rows in by_match.items()}

    goal_rank_rows: list[dict[str, object]] = []
    for goal in goals:
        match_id, timestamp = goal["match_id"], float(goal["event_timestamp_seconds"])
        raw_rows = sorted(by_match[match_id], key=lambda item: float(item["rank_score"]), reverse=True)
        selected_rows = selected_by_match[match_id]
        raw_ranks = [index for index, row in enumerate(raw_rows, start=1) if float(row["best_timestamp_seconds"]) - args.clip_pre_seconds <= timestamp <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds]
        diverse_ranks = [index for index, row in enumerate(selected_rows, start=1) if float(row["best_timestamp_seconds"]) - args.clip_pre_seconds <= timestamp <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds]
        goal_rank_rows.append({"goal_catalog_id": goal["goal_catalog_id"], "match_id": match_id, "goal_timestamp_seconds": f"{timestamp:.3f}", "best_raw_group_rank": raw_ranks[0] if raw_ranks else "", "best_diverse_candidate_rank": diverse_ranks[0] if diverse_ranks else "", "covered_at_max_k": str(bool(diverse_ranks))})

    recall_rows: list[dict[str, object]] = []
    for limit in limits:
        covered = sum(int(row["best_diverse_candidate_rank"] != "" and int(row["best_diverse_candidate_rank"]) <= limit) for row in goal_rank_rows)
        recall_rows.append({"k": limit, "catalog_goals": len(goals), "covered_goals": covered, "recall_at_k": f"{covered / len(goals):.6f}"})

    args.output.mkdir(parents=True)
    write_csv(args.output / "lomo_predictions_v1.csv", predicted_rows, [*fields, "rank_score"])
    write_csv(args.output / "recall_at_k_lomo_v1.csv", recall_rows, ["k", "catalog_goals", "covered_goals", "recall_at_k"])
    write_csv(args.output / "goal_best_rank_lomo_v1.csv", goal_rank_rows, list(goal_rank_rows[0]))
    summary = {"experiment_id": args.output.name, "scope": "somente TRAIN; leave-one-match-out", "groups": len(rows), "positive_groups": int(y.sum()), "matches": len(set(matches)), "features": feature_names, "configuration": configuration, "normalization_by_event": "não aplicada: os pesos do XGBRanker são por grupo de consulta, não por item; evitar semântica ambígua.", "warning": "Comparação de desenvolvimento. Não usar VALIDATION ou TEST para treinar, ajustar parâmetros ou escolher limiares."}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recall_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
