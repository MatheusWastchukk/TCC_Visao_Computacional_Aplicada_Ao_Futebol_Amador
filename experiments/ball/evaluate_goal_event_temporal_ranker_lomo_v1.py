"""Compara rankers temporais em leave-one-match-out, somente TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def select(rows: list[dict[str, object]], limit: int, separation: float) -> list[dict[str, object]]:
    chosen = []
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara rankers temporais LOMO em grupos TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}.")
    rows, fields = read_csv(args.dataset)
    feature_names = [field for field in fields if field not in {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}]
    required = {"match_id", "best_timestamp_seconds", "weak_label"}
    if missing := required - set(fields):
        raise ValueError(f"Dataset sem colunas: {sorted(missing)}")
    goals, goal_fields = read_csv(args.goal_timestamps)
    if any(goal["split"] != "train" for goal in goals):
        raise AssertionError("A avaliação aceita somente TRAIN; TEST permanece fechado.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for goal in goals:
        if goal["annotation_status"] == "completed" and goal["event_timestamp_seconds"]:
            goals_by_match[goal["match_id"]].append(float(goal["event_timestamp_seconds"]))
    x = np.array([[float(row[name]) for name in feature_names] for row in rows])
    y = np.array([int(row["weak_label"]) for row in rows])
    matches = np.array([row["match_id"] for row in rows])
    models = {
        "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260910)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=20260910),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=20260910),
    }
    predictions: dict[str, list[dict[str, object]]] = {name: [] for name in models}
    for held_match in sorted(set(matches)):
        train = matches != held_match
        test = ~train
        for name, factory in models.items():
            model = factory()
            if name == "gradient_boosting":
                model.fit(x[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
            else:
                model.fit(x[train], y[train])
            scores = model.predict_proba(x[test])[:, 1]
            for original, score in zip(np.array(rows, dtype=object)[test], scores):
                predictions[name].append({**original, "score": float(score)})
    result_rows = []
    prediction_rows = []
    evaluated_goals = sum(len(values) for values in goals_by_match.values())
    for name, predicted_rows in predictions.items():
        for row in predicted_rows:
            prediction_rows.append({
                "model": name,
                "group_uid": row["group_uid"],
                "match_id": row["match_id"],
                "best_timestamp_seconds": row["best_timestamp_seconds"],
                "candidate_start_seconds": f"{float(row['best_timestamp_seconds']) - args.clip_pre_seconds:.6f}",
                "candidate_end_seconds": f"{float(row['best_timestamp_seconds']) + args.clip_post_seconds:.6f}",
                "rank_score": f"{float(row['score']):.6f}",
                "weak_label": row["weak_label"],
            })
        for limit in sorted(set(args.candidate_limits)):
            covered = 0
            for match_id, timestamps in goals_by_match.items():
                chosen = select([row for row in predicted_rows if row["match_id"] == match_id], limit, args.min_separation_seconds)
                for timestamp in timestamps:
                    covered += int(any(float(row["best_timestamp_seconds"]) - args.clip_pre_seconds <= timestamp <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds for row in chosen))
            result_rows.append({"model": name, "k": limit, "evaluable_catalog_goals": evaluated_goals, "covered_catalog_goals": covered, "recall_at_k": f"{covered / evaluated_goals:.6f}"})
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "recall_at_k_lomo_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader(); writer.writerows(result_rows)
    with (args.output / "group_predictions_lomo_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    summary = {"experiment_id": "GOAL-EVENT-TEMPORAL-RANKER-EXP-0001", "scope": "somente TRAIN; leave-one-match-out", "groups": len(rows), "positive_groups": int(y.sum()), "features": feature_names, "note": "Recall@K mede cobertura temporal em desenvolvimento; não é resultado final nem confirmação online."}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
