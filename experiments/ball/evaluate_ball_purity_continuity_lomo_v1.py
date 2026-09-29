"""Testa sinais de multiplicidade de caixas e cobertura de trajetória em LOMO TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "results/local_trajectory_dataset_train_v2/temporal_group_dataset_local_trajectory_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_AUDIT = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/manual_review_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PURITY-CONTINUITY-ABLATION-LOMO-EXP-0001"
EXCLUDED = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
NEW_FEATURES = (
    "detections_per_sampled_timestamp",
    "local_path_coverage",
    "local_path_share_of_hits",
    "ball_confidence_mean_to_max",
)
MODEL = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para gravar em {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def with_features(row: dict[str, str]) -> dict[str, float]:
    hits = max(0.0, float(row["hit_count"]))
    samples = max(1.0, float(row["unique_timestamps"]))
    path_obs = max(0.0, float(row["local_trajectory_path_observations"]))
    max_ball = max(0.0, float(row["max_ball_confidence"]))
    mean_ball = max(0.0, float(row["mean_ball_confidence"]))
    return {
        NEW_FEATURES[0]: hits / samples,
        NEW_FEATURES[1]: path_obs / samples,
        NEW_FEATURES[2]: path_obs / max(hits, 1.0),
        NEW_FEATURES[3]: mean_ball / max(max_ball, 1e-8),
    }


def select(rows: list[dict[str, object]], limit: int, gap: float, pre: float, post: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (-float(item["rank_score"]), str(item["group_uid"]))):
        center = float(row["best_timestamp_seconds"])
        if all(abs(center - float(other["best_timestamp_seconds"])) >= gap for other in selected):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def metrics(method: str, k: int, selection: dict[str, list[dict[str, object]]], goals: dict[str, list[float]], pre: float, post: float) -> dict[str, object]:
    covered = sum(
        any(max(0.0, float(row["best_timestamp_seconds"]) - pre) <= goal <= float(row["best_timestamp_seconds"]) + post
            for row in selection[match])
        for match, timestamps in goals.items() for goal in timestamps
    )
    segments = 0
    review_seconds = 0.0
    for rows in selection.values():
        intervals = sorted((max(0.0, float(row["best_timestamp_seconds"]) - pre),
                            float(row["best_timestamp_seconds"]) + post) for row in rows)
        merged: list[list[float]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segments += len(merged)
        review_seconds += sum(end - start for start, end in merged)
    total = sum(map(len, goals.values()))
    return {
        "method": method, "k_per_match": k, "catalog_goals": total,
        "covered_goals": covered, "recall": f"{covered / total:.6f}",
        "selected_candidates": sum(map(len, selection.values())), "merged_segments": segments,
        "unique_review_minutes_per_match": f"{review_seconds / 60 / len(selection):.3f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--manual-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limits", type=int, nargs="+", default=[20, 40, 60, 100])
    parser.add_argument("--gap-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=20.0)
    parser.add_argument("--clip-post-seconds", type=float, default=40.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}")
    rows = read_csv(args.dataset)
    if not rows:
        raise ValueError("Dataset vazio.")
    required = {"match_id", "group_uid", "best_timestamp_seconds", "weak_label", "hit_count", "unique_timestamps",
                "local_trajectory_path_observations", "max_ball_confidence", "mean_ball_confidence"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Dataset sem colunas necessárias: {sorted(missing)}")
    base_features = [field for field in rows[0] if field not in EXCLUDED]
    enriched = [{**row, **{name: f"{value:.8f}" for name, value in with_features(row).items()}} for row in rows]
    feature_sets = {"local_trajectory_baseline": [], **{f"ablate_{name}": [name] for name in NEW_FEATURES}, "all_four": list(NEW_FEATURES)}
    matrices = {
        method: np.asarray([[float(row[name]) for name in [*base_features, *added]] for row in enriched], dtype=np.float32)
        for method, added in feature_sets.items()
    }
    y = np.asarray([int(row["weak_label"]) for row in rows], dtype=np.int64)
    matches = np.asarray([row["match_id"] for row in rows])
    predictions = {name: np.zeros(len(rows)) for name in feature_sets}
    row_indexes: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        row_indexes[row["match_id"]][row["group_uid"]] = index
    for held_match in sorted(set(matches)):
        train_mask, test_mask = matches != held_match, matches == held_match
        for name, matrix in matrices.items():
            model = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            model.fit(matrix[train_mask], y[train_mask])
            predictions[name][test_mask] = model.predict_proba(matrix[test_mask])[:, 1]
    goals: dict[str, list[float]] = defaultdict(list)
    goal_rows = read_csv(args.goal_timestamps)
    for row in goal_rows:
        if row.get("split", "train").lower() != "train":
            raise AssertionError("Somente TRAIN é permitido; VALIDATION/TEST permanecem fechados.")
        if row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds"):
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if len(set(matches)) != 22 or sum(map(len, goals.values())) != 130 or set(goals) != set(matches):
        raise ValueError(f"Escopo esperado: 22 partidas/130 gols; recebido {len(set(matches))}/{sum(map(len, goals.values()))}")

    ranking_rows: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    prediction_output: list[dict[str, object]] = []
    for name, scores in predictions.items():
        for index, source in enumerate(rows):
            row = {**source, "rank_score": float(scores[index])}
            ranking_rows[name][source["match_id"]].append(row)
            prediction_output.append({
                "method": name, "match_id": source["match_id"], "group_uid": source["group_uid"],
                "best_timestamp_seconds": source["best_timestamp_seconds"],
                "weak_label": source["weak_label"], "rank_score": f"{float(scores[index]):.8f}",
            })
    comparison: list[dict[str, object]] = []
    selections: dict[tuple[str, int], dict[str, list[dict[str, object]]]] = {}
    for name, by_match in ranking_rows.items():
        for k in args.limits:
            selected = {match: select(match_rows, k, args.gap_seconds, args.clip_pre_seconds, args.clip_post_seconds)
                        for match, match_rows in by_match.items()}
            selections[(name, k)] = selected
            comparison.append(metrics(name, k, selected, goals, args.clip_pre_seconds, args.clip_post_seconds))

    goal_rank_rows: list[dict[str, object]] = []
    for goal in read_csv(args.manual_audit):
        match = goal["match_id"]
        timestamp = float(goal["goal_timestamp_seconds"])
        for method, by_match in ranking_rows.items():
            if match not in by_match:
                for k in args.limits:
                    goal_rank_rows.append({"goal_catalog_id": goal["goal_catalog_id"], "method": method,
                                           "k_per_match": k, "best_candidate_rank": "",
                                           "covered_by_selection": "not_evaluable_no_match_features"})
                continue
            ordered = sorted(by_match[match], key=lambda row: (-float(row["rank_score"]), str(row["group_uid"])))
            best_rank = next((rank for rank, row in enumerate(ordered, 1)
                              if max(0.0, float(row["best_timestamp_seconds"]) - args.clip_pre_seconds) <= timestamp <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds), "")
            for k in args.limits:
                selected = selections[(method, k)][match]
                covered = any(max(0.0, float(row["best_timestamp_seconds"]) - args.clip_pre_seconds) <= timestamp <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds for row in selected)
                goal_rank_rows.append({"goal_catalog_id": goal["goal_catalog_id"], "method": method,
                                       "k_per_match": k, "best_candidate_rank": best_rank,
                                       "covered_by_selection": covered})

    args.output.mkdir(parents=True)
    write_csv(args.output / "recall_at_k_comparison_v1.csv", comparison)
    write_csv(args.output / "lomo_predictions_v1.csv", prediction_output)
    write_csv(args.output / "manual_miss_diagnostic_v1.csv", goal_rank_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 games, 5,372 groups, 130 completed catalog goals; leave-one-match-out",
        "models": "Paired RandomForestClassifier folds (300 trees, min_samples_leaf=2, class_weight=balanced, random_state=42)",
        "baseline_features": base_features, "added_features": list(NEW_FEATURES),
        "ablation_feature_sets": feature_sets,
        "feature_interpretation": {
            "detections_per_sampled_timestamp": "proxy for multiple ball detections among hits near the goal; not a full-frame false-positive count",
            "local_path_coverage": "fraction of sampled timestamps represented in the selected local path",
            "local_path_share_of_hits": "fraction of all near-goal hits retained by the selected local path",
            "ball_confidence_mean_to_max": "mean confidence relative to the peak confidence in the temporal group",
        },
        "selection": {"window_seconds": [args.clip_pre_seconds, args.clip_post_seconds], "gap_seconds": args.gap_seconds, "limits": args.limits},
        "warning": "Exploratory paired TRAIN/LOMO comparison; weak TRAIN labels; added ratios use existing aggregate features and do not access the 12 human annotations. No policy promoted. TEST closed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": args.output.name, "groups": len(rows), "matches": len(set(matches)), "goals": sum(map(len, goals.values())), "results": comparison}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
