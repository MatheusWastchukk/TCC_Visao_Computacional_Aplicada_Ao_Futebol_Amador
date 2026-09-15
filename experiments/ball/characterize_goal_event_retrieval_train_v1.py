"""Caracteriza cobertura, ranking e custo de revisão do pipeline, somente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_diverse(rows: list[dict[str, str]], separation: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in selected):
            continue
        selected.append(row)
    return selected


def covers(row: dict[str, str], goal: float, pre: float, post: float) -> bool:
    center = float(row["best_timestamp_seconds"])
    return center - pre <= goal <= center + post


def main() -> int:
    parser = argparse.ArgumentParser(description="Caracteriza recuperação de eventos em TRAIN sem retreinamento.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["train", "validation"], default="train")
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20, 30, 40, 50, 60, 80, 100])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    groups = read_csv(args.dataset)
    predictions = read_csv(args.rank_predictions)
    if "model" in predictions[0]:
        predictions = [row for row in predictions if row["model"] == "random_forest"]
    catalog = read_csv(args.goal_timestamps)
    if any(row["split"] != args.expected_split for row in catalog):
        raise AssertionError(f"A caracterização aceita exclusivamente {args.expected_split.upper()}.")
    groups_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    predictions_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in groups:
        groups_by_match[row["match_id"]].append(row)
    for row in predictions:
        predictions_by_match[row["match_id"]].append(row)
    goals_by_match: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append((row["goal_catalog_id"], float(row["event_timestamp_seconds"])))
    matches = set(goals_by_match)
    if matches != set(groups_by_match) or matches != set(predictions_by_match):
        raise ValueError("Dataset, previsões e catálogo TRAIN não possuem as mesmas partidas.")
    selected_by_match = {match: select_diverse(rows, args.min_separation_seconds) for match, rows in predictions_by_match.items()}
    windows = [(10.0, 20.0), (15.0, 30.0), (20.0, 40.0)]
    limits = sorted(set(args.candidate_limits))
    goal_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    all_goals = sum(len(values) for values in goals_by_match.values())
    for pre, post in windows:
        window_key = f"pre_{int(pre)}_post_{int(post)}"
        current: list[dict[str, object]] = []
        for match_id, goals in sorted(goals_by_match.items()):
            ranked = selected_by_match[match_id]
            for goal_id, timestamp in goals:
                ranks = [index for index, row in enumerate(ranked, start=1) if covers(row, timestamp, pre, post)]
                best_rank = ranks[0] if ranks else None
                position_percent = ""
                if best_rank:
                    candidate = ranked[best_rank - 1]
                    position_percent = f"{100.0 * (timestamp - (float(candidate['best_timestamp_seconds']) - pre)) / (pre + post):.3f}"
                row = {
                    "clip_pre_seconds": pre,
                    "clip_post_seconds": post,
                    "match_id": match_id,
                    "goal_catalog_id": goal_id,
                    "goal_timestamp_seconds": f"{timestamp:.3f}",
                    "candidate_pool_contains_goal": str(any(covers(group, timestamp, pre, post) for group in groups_by_match[match_id])),
                    "diverse_pool_contains_goal": str(best_rank is not None),
                    "best_diverse_rank": best_rank or "",
                    "goal_position_percent_in_best_clip": position_percent,
                }
                current.append(row)
                goal_rows.append(row)
        pool = sum(row["candidate_pool_contains_goal"] == "True" for row in current)
        diverse_pool = sum(row["diverse_pool_contains_goal"] == "True" for row in current)
        recalls: dict[int, float] = {}
        for limit in limits:
            covered = sum(bool(row["best_diverse_rank"]) and int(row["best_diverse_rank"]) <= limit for row in current)
            recall = covered / all_goals
            recalls[limit] = recall
            curve_rows.append({
                "clip_pre_seconds": pre, "clip_post_seconds": post, "k": limit,
                "catalog_goals": all_goals, "candidate_pool_goals": pool,
                "diverse_pool_goals": diverse_pool, "covered_goals": covered,
                "recall_at_k": f"{recall:.6f}",
                "review_seconds_per_match": int(limit * (pre + post)),
                "review_minutes_per_match": f"{limit * (pre + post) / 60.0:.2f}",
            })
        for target in (0.50, 0.75, 0.90, 0.95, 1.00):
            k_required = next((limit for limit in limits if recalls[limit] >= target), "")
            target_rows.append({"clip_pre_seconds": pre, "clip_post_seconds": post, "target_recall": f"{target:.2f}", "k_required_within_evaluated_curve": k_required})
        for match_id, goals in sorted(goals_by_match.items()):
            ranks = [int(row["best_diverse_rank"]) for row in current if row["match_id"] == match_id and row["best_diverse_rank"]]
            recoverable = len(ranks)
            per_match_rows.append({
                "clip_pre_seconds": pre, "clip_post_seconds": post, "match_id": match_id,
                "catalog_goals": len(goals), "recoverable_goals_in_diverse_pool": recoverable,
                "unrecoverable_goals_in_diverse_pool": len(goals) - recoverable,
                "k_min_all_recoverable_goals": max(ranks) if ranks else "",
                "median_recoverable_goal_rank": f"{float(np.median(ranks)):.2f}" if ranks else "",
            })
        numeric_ranks = [int(row["best_diverse_rank"]) for row in current if row["best_diverse_rank"]]
        positions = [float(row["goal_position_percent_in_best_clip"]) for row in current if row["goal_position_percent_in_best_clip"]]
        summaries[window_key] = {
            "candidate_pool_recall": pool / all_goals,
            "diverse_pool_recall": diverse_pool / all_goals,
            "rank_median": float(np.median(numeric_ranks)) if numeric_ranks else None,
            "rank_p75": float(np.percentile(numeric_ranks, 75)) if numeric_ranks else None,
            "rank_p90": float(np.percentile(numeric_ranks, 90)) if numeric_ranks else None,
            "goal_position_percent_median": float(np.median(positions)) if positions else None,
            "max_candidates_in_a_match": max(len(rows) for rows in selected_by_match.values()),
        }
    summary = {
        "experiment_id": args.output.name,
        "scope": f"somente {args.expected_split.upper()}; caracterização sem retreinamento",
        "catalog_goals": all_goals,
        "min_separation_seconds": args.min_separation_seconds,
        "candidate_limits": limits,
        "windows": summaries,
        "warning": "Esta análise mede teto e trade-off sem retreinamento. TEST não foi acessado e o resultado não autoriza alterar a configuração avaliada.",
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "goal_rank_and_clip_position_v1.csv", goal_rows)
    write_csv(args.output / "recall_review_curve_v1.csv", curve_rows)
    write_csv(args.output / "per_match_k_min_v1.csv", per_match_rows)
    write_csv(args.output / "coverage_targets_v1.csv", target_rows)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
