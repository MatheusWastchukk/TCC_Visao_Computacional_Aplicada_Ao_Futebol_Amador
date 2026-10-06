"""TRAIN-only candidate selection under an explicit unique-video-minute budget."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_complementarity_matrix_v1 import (
    BASE_SCORES, EXPERTS, ROOT, RRF_OFFSET, read_csv, sha256, write_csv,
)

DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/REVIEW-BUDGET-SELECTION-EXP-0001"
BUDGET_MINUTES = (20, 25, 30, 35, 40, 45, 50, 60)


def interval_for(row: dict[str, object]) -> tuple[float, float]:
    return float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])


def added_seconds(interval: tuple[float, float], merged: list[tuple[float, float]]) -> float:
    start, end = interval
    overlap = sum(max(0.0, min(end, right) - max(start, left)) for left, right in merged)
    return max(0.0, end - start - overlap)


def insert_interval(interval: tuple[float, float], merged: list[tuple[float, float]]) -> list[tuple[float, float]]:
    start, end = interval
    out: list[tuple[float, float]] = []
    inserted = False
    for left, right in merged:
        if right < start:
            out.append((left, right))
        elif end < left:
            if not inserted:
                out.append((start, end))
                inserted = True
            out.append((left, right))
        else:
            start, end = min(start, left), max(end, right)
    if not inserted:
        out.append((start, end))
    return out


def interval_cost(merged: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merged)


def select_rank_budget(rows: list[dict[str, object]], score_name: str, budget_seconds: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    merged: list[tuple[float, float]] = []
    for row in sorted(rows, key=lambda r: (-float(r[score_name]), str(r["group_uid"]))):
        interval = interval_for(row)
        delta = added_seconds(interval, merged)
        if interval_cost(merged) + delta <= budget_seconds + 1e-7:
            selected.append(row)
            merged = insert_interval(interval, merged)
    return selected


def select_score_per_second(rows: list[dict[str, object]], score_name: str, budget_seconds: float) -> list[dict[str, object]]:
    pending = list(rows)
    selected: list[dict[str, object]] = []
    merged: list[tuple[float, float]] = []
    while pending:
        current_cost = interval_cost(merged)
        free: list[dict[str, object]] = []
        best: dict[str, object] | None = None
        best_ratio = -1.0
        best_delta = 0.0
        for row in pending:
            delta = added_seconds(interval_for(row), merged)
            if delta <= 1e-7:
                free.append(row)
                continue
            if current_cost + delta > budget_seconds + 1e-7:
                continue
            ratio = float(row[score_name]) / delta
            if ratio > best_ratio or (ratio == best_ratio and str(row["group_uid"]) < str(best["group_uid"])):
                best, best_ratio, best_delta = row, ratio, delta
        if free:
            # These windows add no delivered seconds; add them together without changing
            # the union, then recompute marginal costs for the remaining candidates.
            free_ids = {str(row["group_uid"]) for row in free}
            selected.extend(sorted(free, key=lambda r: (-float(r[score_name]), str(r["group_uid"]))))
            pending = [row for row in pending if str(row["group_uid"]) not in free_ids]
            continue
        if best is None:
            break
        selected.append(best)
        merged = insert_interval(interval_for(best), merged)
        best_id = str(best["group_uid"])
        pending = [row for row in pending if str(row["group_uid"]) != best_id]
    return selected


def rrf_score(rows: list[dict[str, object]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for source in (*BASE_SCORES, "net_motion"):
        ordered = sorted(rows, key=lambda r: (-float(r[source]), str(r["group_uid"])))
        for rank, row in enumerate(ordered, 1):
            scores[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
    return scores


def measure(rows: list[dict[str, object]], goals: list[dict[str, str]]) -> tuple[int, float, int]:
    hit = sum(
        any(float(row["delivery_start_seconds"]) <= float(goal["event_timestamp_seconds"]) <= float(row["delivery_end_seconds"]) for row in rows)
        for goal in goals
    )
    merged: list[tuple[float, float]] = []
    for row in rows:
        merged = insert_interval(interval_for(row), merged)
    return hit, interval_cost(merged) / 60.0, len(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")

    pool = read_csv(args.candidate_scores)
    pool_by_key = {(row["match_id"], row["group_uid"]): row for row in pool}
    expert_maps = {
        name: {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in read_csv(path)}
        for name, path in EXPERTS.items()
    }
    common = set(pool_by_key)
    for scores in expert_maps.values():
        common &= set(scores)
    if len(common) != 5372:
        raise ValueError(f"Expected frozen common TRAIN population 5,372; found {len(common)}")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(common):
        row: dict[str, object] = dict(pool_by_key[key])
        for source, scores in expert_maps.items():
            row[source] = scores[key]
        by_match[key[0]].append(row)
    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Only TRAIN catalog rows are allowed")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or set(by_match) != set(goals) or sum(map(len, goals.values())) != 130:
        raise ValueError("Expected exactly 22 TRAIN matches and 130 completed goals")

    methods = (
        ("TEMPORAL_RANK_FITS_BUDGET", "rank_score", "rank"),
        ("RRF4_NET_RANK_FITS_BUDGET", "rrf4_net_score", "rank"),
        ("META_SCORE_PER_ADDED_SECOND", "rank_score", "ratio"),
        ("RRF4_NET_SCORE_PER_ADDED_SECOND", "rrf4_net_score", "ratio"),
    )
    aggregate: list[dict[str, object]] = []
    per_match: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for match, raw_rows in by_match.items():
        rrf = rrf_score(raw_rows)
        rows = []
        for source in raw_rows:
            row = dict(source)
            row["rrf4_net_score"] = rrf[str(row["group_uid"])]
            rows.append(row)
        for budget in BUDGET_MINUTES:
            for method, score, policy in methods:
                chosen = (
                    select_rank_budget(rows, score, budget * 60.0)
                    if policy == "rank" else select_score_per_second(rows, score, budget * 60.0)
                )
                covered, minutes, segments = measure(chosen, goals[match])
                per_match.append({
                    "method": method, "budget_minutes_per_match": budget, "match_id": match,
                    "catalog_goals": len(goals[match]), "covered_goals": covered,
                    "selected_candidates": len(chosen), "unique_review_minutes": f"{minutes:.3f}",
                    "merged_segments": segments,
                })
                for rank, row in enumerate(chosen, 1):
                    selected_rows.append({
                        "method": method, "budget_minutes_per_match": budget, "match_id": match,
                        "selection_order": rank, "group_uid": row["group_uid"],
                        "utility_score": f"{float(row[score]):.10f}",
                        "delivery_start_seconds": row["delivery_start_seconds"],
                        "delivery_end_seconds": row["delivery_end_seconds"],
                    })

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_match:
        grouped[(str(row["method"]), str(row["budget_minutes_per_match"]))].append(row)
    for (method, budget), rows in sorted(grouped.items()):
        n_goals = sum(int(row["catalog_goals"]) for row in rows)
        covered = sum(int(row["covered_goals"]) for row in rows)
        aggregate.append({
            "method": method, "budget_minutes_per_match": int(budget),
            "catalog_goals": n_goals, "covered_goals": covered,
            "recall": f"{covered / n_goals:.6f}",
            "selected_candidates": sum(int(row["selected_candidates"]) for row in rows),
            "actual_unique_review_minutes_per_match": f"{sum(float(row['unique_review_minutes']) for row in rows) / len(rows):.3f}",
            "merged_segments": sum(int(row["merged_segments"]) for row in rows),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "review_budget_summary_v1.csv", aggregate)
    write_csv(args.output / "review_budget_by_match_v1.csv", per_match)
    write_csv(args.output / "review_budget_selected_candidates_v1.csv", selected_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matches; 5,372 common groups; 130 completed goals; no VALIDATION or TEST",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]],
        "budgets": list(BUDGET_MINUTES), "window": "frozen delivery intervals (20 seconds before and 40 seconds after center)",
        "selection": "Rank-fit selects in frozen score order and skips any candidate that exceeds the unique-minute cap. Greedy policies dynamically maximize score divided by additional unique seconds; zero-cost overlapping windows are added without increasing the budget.",
        "methods": [row[0] for row in methods],
        "interpretation": "Exploratory post-hoc evaluation on TRAIN/LOMO. Candidate selection uses scores/intervals only; goal labels are used only after selection to measure coverage. Not independent validation or promotion.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"methods": len(methods), "budgets": len(BUDGET_MINUTES), "aggregate_rows": len(aggregate), "per_match_rows": len(per_match), "selected_rows": len(selected_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
