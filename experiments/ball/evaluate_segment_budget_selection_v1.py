"""TRAIN-only selection of temporally clustered candidate windows as review segments."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from build_complementarity_matrix_v1 import (
    BASE_SCORES, EXPERTS, GAP_SECONDS, ROOT, RRF_OFFSET,
    read_csv, select_with_gap, sha256, write_csv,
)

DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/SEGMENT-BUDGET-SELECTION-EXP-0001"
K_PER_SOURCE = 60
BUDGETS = (20, 30, 40, 50, 60)
FEATURES = ("max_normalized_score", "mean_top3_normalized_score", "support_count", "n_methods", "n_candidates", "source_diversity", "best_rank_score")


def center(row: dict[str, object]) -> float:
    return float(row["delivery_start_seconds"]) + 20.0


def rrf_rank(rows: list[dict[str, object]], sources: tuple[str, ...]) -> list[dict[str, object]]:
    score: dict[str, float] = defaultdict(float)
    for source in sources:
        ordered = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        for rank, row in enumerate(ordered, 1):
            score[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
    return sorted(rows, key=lambda row: (-score[str(row["group_uid"])], str(row["group_uid"])))


def cluster(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(candidates, key=lambda row: (float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"]), str(row["group_uid"])))
    result: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    start = end = 0.0
    for row in ordered:
        a, b = float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])
        if members and a > end:
            result.append(make_segment(start, end, members))
            members = []
        if not members:
            start, end = a, b
        else:
            end = max(end, b)
        members.append(row)
    if members:
        result.append(make_segment(start, end, members))
    return result


def make_segment(start: float, end: float, members: list[dict[str, object]]) -> dict[str, object]:
    candidate_scores = sorted((float(row["normalized_score"]) for row in members), reverse=True)
    support: dict[str, int] = defaultdict(int)
    best_rank = math.inf
    for row in members:
        for source in row["support_sources"]:
            support[str(source)] += 1
        best_rank = min(best_rank, float(row["best_rank"]))
    n_support = sum(support.values())
    entropy = -sum((count / n_support) * math.log(count / n_support) for count in support.values()) if n_support else 0.0
    diversity = entropy / math.log(len(support)) if len(support) > 1 else 0.0
    return {
        "segment_id": f"seg_{start:.3f}_{end:.3f}", "segment_start_seconds": start,
        "segment_end_seconds": end, "duration_seconds": end - start,
        "members": list(members), "max_normalized_score": candidate_scores[0],
        "mean_top3_normalized_score": sum(candidate_scores[:3]) / min(3, len(candidate_scores)),
        "support_count": n_support, "n_methods": len(support), "n_candidates": len(members),
        "source_diversity": diversity, "best_rank_score": 1.0 / (RRF_OFFSET + best_rank),
        "support_sources": sorted(support),
    }


def select_segments(segments: list[dict[str, object]], feature: str, budget_seconds: float) -> list[dict[str, object]]:
    ordered = sorted(segments, key=lambda row: (-float(row[feature]), float(row["segment_start_seconds"]), str(row["segment_id"])))
    chosen: list[dict[str, object]] = []
    total = 0.0
    for segment in ordered:
        duration = float(segment["duration_seconds"])
        if total + duration <= budget_seconds + 1e-7:
            chosen.append(segment)
            total += duration
    return chosen


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
    expert_maps = {name: {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in read_csv(path)} for name, path in EXPERTS.items()}
    common = set(pool_by_key)
    for values in expert_maps.values():
        common &= set(values)
    if len(common) != 5372:
        raise ValueError(f"Expected 5,372 common TRAIN groups; found {len(common)}")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(common):
        row: dict[str, object] = dict(pool_by_key[key])
        for name, values in expert_maps.items():
            row[name] = values[key]
        by_match[key[0]].append(row)
    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Only TRAIN catalog rows are allowed")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or set(by_match) != set(goals) or sum(map(len, goals.values())) != 130:
        raise ValueError("Expected exactly 22 TRAIN matches and 130 completed goals")

    method_sources = {
        "temporal": "rank_score", "visual": "visual_score_goal", "fusion": "fusion_score_goal",
        "local_trajectory": "local_trajectory", "shot_motion": "shot_motion", "net_motion": "net_motion",
    }
    segments_by_match: dict[str, list[dict[str, object]]] = {}
    for match, rows in by_match.items():
        rankings = {name: sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"]))) for name, source in method_sources.items()}
        rankings["RRF3"] = rrf_rank(rows, BASE_SCORES)
        rankings["RRF4_net"] = rrf_rank(rows, (*BASE_SCORES, "net_motion"))
        membership: dict[str, set[str]] = defaultdict(set)
        rank_maps: dict[str, dict[str, int]] = {}
        for method, ordered in rankings.items():
            rank_maps[method] = {str(row["group_uid"]): rank for rank, row in enumerate(ordered, 1)}
            for row in select_with_gap(ordered, K_PER_SOURCE):
                membership[str(row["group_uid"])].add(method)
        row_by_uid = {str(row["group_uid"]): row for row in rows}
        candidates: list[dict[str, object]] = []
        for uid, sources in membership.items():
            row = dict(row_by_uid[uid])
            best_rank = min(rank_maps[method][uid] for method in sources)
            row["support_sources"] = sorted(sources)
            row["normalized_score"] = 1.0 / (RRF_OFFSET + best_rank)
            row["best_rank"] = best_rank
            candidates.append(row)
        segments = cluster(candidates)
        for segment in segments:
            segment["match_id"] = match
        segments_by_match[match] = segments

    summary: list[dict[str, object]] = []
    per_match: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for feature in FEATURES:
        for budget in BUDGETS:
            total_goals = covered_goals = total_segments = total_candidates = 0
            total_minutes = 0.0
            matches_with_gain = 0
            for match, segments in segments_by_match.items():
                chosen = select_segments(segments, feature, budget * 60.0)
                covered = sum(any(float(seg["segment_start_seconds"]) <= float(goal["event_timestamp_seconds"]) <= float(seg["segment_end_seconds"]) for seg in chosen) for goal in goals[match])
                total_goals += len(goals[match])
                covered_goals += covered
                total_segments += len(chosen)
                total_candidates += sum(len(seg["members"]) for seg in chosen)
                actual_minutes = sum(float(seg["duration_seconds"]) for seg in chosen) / 60.0
                total_minutes += actual_minutes
                per_match.append({
                    "feature_ranker": feature, "budget_minutes_per_match": budget,
                    "match_id": match, "catalog_goals": len(goals[match]),
                    "covered_goals": covered, "selected_segments": len(chosen),
                    "selected_candidate_count": sum(len(seg["members"]) for seg in chosen),
                    "unique_review_minutes": f"{actual_minutes:.3f}",
                })
                for rank, segment in enumerate(chosen, 1):
                    selected_rows.append({
                        "feature_ranker": feature, "budget_minutes_per_match": budget,
                        "match_id": match, "selection_order": rank,
                        "segment_id": segment["segment_id"],
                        "segment_start_seconds": f"{float(segment['segment_start_seconds']):.3f}",
                        "segment_end_seconds": f"{float(segment['segment_end_seconds']):.3f}",
                        "duration_seconds": f"{float(segment['duration_seconds']):.3f}",
                        "n_candidates": segment["n_candidates"], "support_count": segment["support_count"],
                        "n_methods": segment["n_methods"], "source_diversity": f"{float(segment['source_diversity']):.6f}",
                        "max_normalized_score": f"{float(segment['max_normalized_score']):.8f}",
                        "mean_top3_normalized_score": f"{float(segment['mean_top3_normalized_score']):.8f}",
                        "best_rank_score": f"{float(segment['best_rank_score']):.8f}",
                        "support_sources": ";".join(segment["support_sources"]),
                    })
            summary.append({
                "feature_ranker": feature, "budget_minutes_per_match": budget,
                "catalog_goals": total_goals, "covered_goals": covered_goals,
                "recall": f"{covered_goals / total_goals:.6f}",
                "selected_segments": total_segments,
                "selected_candidate_count": total_candidates,
                "actual_unique_review_minutes_per_match": f"{total_minutes / len(by_match):.3f}",
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "segment_budget_summary_v1.csv", summary)
    write_csv(args.output / "segment_budget_by_match_v1.csv", per_match)
    write_csv(args.output / "segment_budget_selected_segments_v1.csv", selected_rows)
    meta = {
        "experiment_id": args.output.name, "scope": "TRAIN only; 22 matches; 5,372 common candidate groups; 130 catalog goals; no VALIDATION or TEST",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]],
        "pool": "Union of per-source Top-60 candidate windows (ranked by temporal, visual, fusion, three LOMO experts, RRF3, RRF4+net); 45-second gap applied within each source",
        "segmenting": "Candidate delivery intervals are clustered when they overlap; segment score features aggregate normalized rank support across candidates and sources. Budget cost is duration of selected segment intervals.",
        "budgets_minutes_per_match": list(BUDGETS), "feature_rankers": list(FEATURES),
        "interpretation": "Exploratory post-hoc TRAIN/LOMO segment-selection comparison. No validation or test data used and no promotion decision.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matches": len(by_match), "segments_in_union": sum(map(len, segments_by_match.values())), "summary_rows": len(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
