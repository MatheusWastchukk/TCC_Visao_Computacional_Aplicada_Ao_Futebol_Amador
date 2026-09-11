"""Avalia ranking de grupos de hits com leave-one-match-out, somente TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "hit_count", "duration_seconds", "max_candidate_score", "mean_candidate_score",
    "max_ball_confidence", "mean_ball_confidence", "max_goal_confidence", "mean_goal_confidence",
    "min_normalized_distance", "mean_normalized_distance", "inside_goal_hit_fraction",
]


@dataclass(frozen=True)
class Hit:
    match_id: str
    timestamp: float
    candidate_score: float
    ball_confidence: float
    goal_confidence: float
    normalized_distance: float
    inside_goal: bool


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def require_fields(fields: list[str], required: set[str], label: str) -> None:
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{label} sem colunas obrigatórias: {sorted(missing)}")


def group_hits(hits: list[Hit], merge_gap_seconds: float, minimum_hits: int) -> list[list[Hit]]:
    groups: list[list[Hit]] = []
    for hit in sorted(hits, key=lambda item: item.timestamp):
        if not groups or hit.timestamp - groups[-1][-1].timestamp > merge_gap_seconds:
            groups.append([hit])
        else:
            groups[-1].append(hit)
    return [group for group in groups if len(group) >= minimum_hits]


def features(group: list[Hit]) -> dict[str, float]:
    scores = [hit.candidate_score for hit in group]
    balls = [hit.ball_confidence for hit in group]
    goals = [hit.goal_confidence for hit in group]
    distances = [hit.normalized_distance for hit in group]
    return {
        "hit_count": float(len(group)),
        "duration_seconds": group[-1].timestamp - group[0].timestamp,
        "max_candidate_score": max(scores),
        "mean_candidate_score": float(np.mean(scores)),
        "max_ball_confidence": max(balls),
        "mean_ball_confidence": float(np.mean(balls)),
        "max_goal_confidence": max(goals),
        "mean_goal_confidence": float(np.mean(goals)),
        "min_normalized_distance": min(distances),
        "mean_normalized_distance": float(np.mean(distances)),
        "inside_goal_hit_fraction": float(np.mean([hit.inside_goal for hit in group])),
    }


def best_timestamp(group: list[Hit]) -> float:
    return max(group, key=lambda hit: hit.candidate_score).timestamp


def select_rows(rows: list[dict[str, object]], limit: int, separation_seconds: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation_seconds for other in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia ranking de grupos com leave-one-match-out em dados TRAIN."
    )
    parser.add_argument("--hit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--merge-gap-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-hits", type=int, default=2)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    parser.add_argument(
        "--positive-radius-seconds",
        type=float,
        default=None,
        help="Raio temporal para rótulo fraco positivo; omitido usa a janela completa do clipe.",
    )
    parser.add_argument("--random-state", type=int, default=20260910)
    args = parser.parse_args()
    if min(args.candidate_limits, default=0) < 1 or (
        args.positive_radius_seconds is not None and args.positive_radius_seconds < 0
    ):
        raise ValueError("candidate-limits deve conter inteiros positivos.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}.")

    hits_by_match: dict[str, list[Hit]] = defaultdict(list)
    for path in args.hit_files:
        rows, fields = read_csv(path)
        require_fields(
            fields,
            {"match_id", "timestamp_seconds", "candidate_score", "ball_confidence", "goal_confidence", "normalized_distance", "ball_inside_expanded_goal"},
            f"Hits em {path}",
        )
        for row in rows:
            hits_by_match[row["match_id"]].append(Hit(
                match_id=row["match_id"],
                timestamp=float(row["timestamp_seconds"]),
                candidate_score=float(row["candidate_score"]),
                ball_confidence=float(row["ball_confidence"]),
                goal_confidence=float(row["goal_confidence"]),
                normalized_distance=float(row["normalized_distance"]),
                inside_goal=row["ball_inside_expanded_goal"] == "True",
            ))

    goals, goal_fields = read_csv(args.goal_timestamps)
    require_fields(goal_fields, {"goal_catalog_id", "match_id", "split", "event_timestamp_seconds", "annotation_status"}, "Timestamps")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in goals:
        if row["split"] != "train":
            raise AssertionError("A avaliação aceita somente TRAIN; TEST permanece fechado.")
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    grouped_by_match = {
        match_id: group_hits(hits, args.merge_gap_seconds, args.minimum_hits)
        for match_id, hits in hits_by_match.items()
    }
    group_rows: list[dict[str, object]] = []
    for match_id, groups in grouped_by_match.items():
        for group_index, group in enumerate(groups, start=1):
            best = best_timestamp(group)
            start = max(0.0, best - args.clip_pre_seconds)
            end = best + args.clip_post_seconds
            is_positive = (
                any(abs(best - goal) <= args.positive_radius_seconds for goal in goals_by_match[match_id])
                if args.positive_radius_seconds is not None
                else any(start <= goal <= end for goal in goals_by_match[match_id])
            )
            row: dict[str, object] = {
                "group_uid": f"{match_id}::group_{group_index:04d}",
                "match_id": match_id,
                "best_timestamp_seconds": best,
                "candidate_start_seconds": start,
                "candidate_end_seconds": end,
                "covers_catalog_goal": is_positive,
            }
            row.update(features(group))
            group_rows.append(row)
    if not group_rows:
        raise ValueError("Nenhum grupo elegível foi reconstruído.")

    for held_match in sorted(grouped_by_match):
        train_rows = [row for row in group_rows if row["match_id"] != held_match]
        held_rows = [row for row in group_rows if row["match_id"] == held_match]
        x_train = np.array([[float(row[feature]) for feature in FEATURES] for row in train_rows])
        y_train = np.array([bool(row["covers_catalog_goal"]) for row in train_rows])
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_state,
        )
        model.fit(x_train, y_train)
        x_held = np.array([[float(row[feature]) for feature in FEATURES] for row in held_rows])
        probabilities = model.predict_proba(x_held)[:, 1]
        for row, probability in zip(held_rows, probabilities):
            row["rank_score"] = float(probability)

    evaluated_goals = sum(len(values) for values in goals_by_match.values())
    summary_rows: list[dict[str, object]] = []
    for limit in sorted(set(args.candidate_limits)):
        covered = 0
        selected_count = 0
        for match_id, match_goals in goals_by_match.items():
            selected = select_rows(
                [row for row in group_rows if row["match_id"] == match_id],
                limit,
                args.min_separation_seconds,
            )
            selected_count += len(selected)
            for goal in match_goals:
                covered += int(any(
                    float(row["candidate_start_seconds"]) <= goal <= float(row["candidate_end_seconds"])
                    for row in selected
                ))
        summary_rows.append({
            "max_candidates_per_match": limit,
            "selected_candidate_windows": selected_count,
            "evaluable_catalog_goals": evaluated_goals,
            "covered_catalog_goals": covered,
            "candidate_window_recall": f"{covered / evaluated_goals:.6f}",
        })

    args.output.mkdir(parents=True, exist_ok=False)
    serialized_groups = [
        {key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()}
        for row in group_rows
    ]
    write_csv(args.output / "group_predictions_lomo_v1.csv", serialized_groups)
    write_csv(args.output / "coverage_by_capacity_lomo_v1.csv", summary_rows)
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda item: item[1], reverse=True)
    summary = {
        "experiment_id": "GOAL-EVENT-GROUP-RANKER-LOMO-V1",
        "scope": "somente TRAIN; leave-one-match-out",
        "evaluable_catalog_goals": evaluated_goals,
        "groups": len(group_rows),
        "positive_groups": sum(bool(row["covers_catalog_goal"]) for row in group_rows),
        "positive_radius_seconds": args.positive_radius_seconds,
        "feature_importance_last_fold": [{"feature": name, "importance": round(float(value), 6)} for name, value in importances],
        "note": "Ranking supervisionado avaliado por partida deixada de fora. É experimento de desenvolvimento TRAIN, não avaliação final nem confirmação online.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
