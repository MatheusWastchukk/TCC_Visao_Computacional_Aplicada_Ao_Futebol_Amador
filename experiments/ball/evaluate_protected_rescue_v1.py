"""Evaluate additive rescue lists while protecting the full temporal Top-60 on TRAIN."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_complementarity_matrix_v1 import (
    BASE_SCORES,
    EXPERTS,
    GAP_SECONDS,
    ROOT,
    RRF_OFFSET,
    merged_cost,
    read_csv,
    select_with_gap,
    sha256,
    write_csv,
)


DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/PROTECTED-RESCUE-EXP-0001"
RESCUE_BUDGETS = (5, 10, 15, 20)
RESCUE_SOURCES = (
    "fusion_score_goal",
    "net_motion",
    "RRF3_temporal_visual_fusion",
    "RRF4_plus_net_motion",
)
OUTPUT_FILES = {
    "protected_rescue_summary_v1.csv",
    "protected_rescue_by_match_v1.csv",
    "protected_rescue_selected_candidates_v1.csv",
    "protected_rescue_goal_coverage_v1.csv",
    "summary_v1.json",
}


def center(row: dict[str, object]) -> float:
    return float(row["delivery_start_seconds"]) + 20.0


def covered_for_match(rows: list[dict[str, object]], goals: list[dict[str, str]]) -> set[str]:
    intervals = [(float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in rows]
    return {
        goal["goal_catalog_id"]
        for goal in goals
        if any(start <= float(goal["event_timestamp_seconds"]) <= end for start, end in intervals)
    }


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
    if len(pool_by_key) != len(pool):
        raise ValueError("Duplicate match_id/group_uid in base candidate scores.")

    expert_maps = {
        name: {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in read_csv(path)}
        for name, path in EXPERTS.items()
    }
    common = set(pool_by_key)
    for score_map in expert_maps.values():
        common &= set(score_map)
    if len(common) != 5372:
        raise ValueError(f"Expected the frozen common TRAIN intersection of 5,372 groups; found {len(common)}.")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(common):
        row: dict[str, object] = dict(pool_by_key[key])
        for source, score_map in expert_maps.items():
            row[source] = score_map[key]
        row["_key"] = f"{key[0]}|{key[1]}"
        by_match[key[0]].append(row)

    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Only TRAIN catalog goals may be read.")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or set(by_match) != set(goals) or sum(map(len, goals.values())) != 130:
        raise ValueError("Expected exactly 22 TRAIN matches and 130 completed catalog goals.")

    ranked: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    for match, rows in by_match.items():
        for source in (*BASE_SCORES, *EXPERTS):
            ranked[source][match] = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        reciprocal: dict[str, float] = defaultdict(float)
        for source in BASE_SCORES:
            for rank, row in enumerate(ranked[source][match], start=1):
                reciprocal[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
        ranked["RRF3_temporal_visual_fusion"][match] = sorted(
            rows, key=lambda row: (-reciprocal[str(row["group_uid"])], str(row["group_uid"]))
        )
        reciprocal_net = defaultdict(float, reciprocal)
        for rank, row in enumerate(ranked["net_motion"][match], start=1):
            reciprocal_net[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
        ranked["RRF4_plus_net_motion"][match] = sorted(
            rows, key=lambda row: (-reciprocal_net[str(row["group_uid"])], str(row["group_uid"]))
        )

    baseline = {match: select_with_gap(ranked["rank_score"][match], 60) for match in by_match}
    baseline_covered = {
        match: covered_for_match(baseline[match], goals[match]) for match in by_match
    }
    baseline_segments, baseline_minutes = merged_cost(baseline)
    summary_rows: list[dict[str, object]] = []
    by_match_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for source in RESCUE_SOURCES:
        for budget in RESCUE_BUDGETS:
            selected: dict[str, list[dict[str, object]]] = {}
            actual_rescues: dict[str, list[dict[str, object]]] = {}
            for match in by_match:
                chosen = list(baseline[match])
                chosen_keys = {str(row["_key"]) for row in chosen}
                added: list[dict[str, object]] = []
                for row in ranked[source][match]:
                    key = str(row["_key"])
                    if key in chosen_keys or any(abs(center(row) - center(other)) < GAP_SECONDS for other in chosen):
                        continue
                    chosen.append(row)
                    chosen_keys.add(key)
                    added.append(row)
                    if len(added) == budget:
                        break
                selected[match] = chosen
                actual_rescues[match] = added
                for row in added:
                    selected_rows.append({
                        "source": source,
                        "requested_rescue_budget": budget,
                        "match_id": match,
                        "group_uid": row["group_uid"],
                        "delivery_start_seconds": row["delivery_start_seconds"],
                        "delivery_end_seconds": row["delivery_end_seconds"],
                        "source_score": row[source] if source in row else "",
                    })

            selected_segments, selected_minutes = merged_cost(selected)
            total_baseline_goals = 0
            total_selected_goals = 0
            rescued_goals = 0
            per_match_rescues = 0
            per_match_budget_rows: list[dict[str, object]] = []
            for match in by_match:
                new_covered = covered_for_match(selected[match], goals[match])
                base_covered = baseline_covered[match]
                gained = new_covered - base_covered
                total_baseline_goals += len(base_covered)
                total_selected_goals += len(new_covered)
                rescued_goals += len(gained)
                per_match_rescues += len(actual_rescues[match])
                per_match_budget_rows.append({
                    "source": source,
                    "requested_rescue_budget": budget,
                    "match_id": match,
                    "base_top60_candidates": len(baseline[match]),
                    "actual_rescue_candidates": len(actual_rescues[match]),
                    "base_goals_covered": len(base_covered),
                    "protected_goals_covered": len(new_covered),
                    "incremental_goals": len(gained),
                    "incremental_minutes": f"{merged_cost({match: selected[match]})[1] - merged_cost({match: baseline[match]})[1]:.3f}",
                    "segment_delta": merged_cost({match: selected[match]})[0] - merged_cost({match: baseline[match]})[0],
                })
                for goal in goals[match]:
                    timestamp = float(goal["event_timestamp_seconds"])
                    base_hit = goal["goal_catalog_id"] in base_covered
                    selected_hit = any(
                        float(row["delivery_start_seconds"]) <= timestamp <= float(row["delivery_end_seconds"])
                        for row in selected[match]
                    )
                    coverage_rows.append({
                        "source": source,
                        "requested_rescue_budget": budget,
                        "match_id": match,
                        "goal_catalog_id": goal["goal_catalog_id"],
                        "baseline_top60_covered": base_hit,
                        "protected_rescue_covered": selected_hit,
                    })
            by_match_rows.extend(per_match_budget_rows)
            delta_minutes = selected_minutes - baseline_minutes
            delta_segments = selected_segments - baseline_segments
            summary_rows.append({
                "source": source,
                "requested_rescue_budget_per_match": budget,
                "baseline_goals": total_baseline_goals,
                "protected_goals": total_selected_goals,
                "incremental_goals": rescued_goals,
                "actual_rescue_candidates": per_match_rescues,
                "mean_rescue_candidates_per_match": f"{per_match_rescues / len(by_match):.3f}",
                "delta_unique_review_minutes_per_match": f"{delta_minutes:.3f}",
                "delta_merged_segments": delta_segments,
                "incremental_goals_per_extra_minute": f"{(rescued_goals / delta_minutes):.3f}" if delta_minutes > 0 else "",
                "base_k": 60,
                "total_candidate_budget_max_per_match": 60 + budget,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "protected_rescue_summary_v1.csv", summary_rows)
    write_csv(args.output / "protected_rescue_by_match_v1.csv", by_match_rows)
    write_csv(args.output / "protected_rescue_selected_candidates_v1.csv", selected_rows)
    write_csv(args.output / "protected_rescue_goal_coverage_v1.csv", coverage_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matches; 5,372 common candidate groups; 130 completed catalog goals; input scores OOF by match",
        "protocol": {
            "baseline": "full temporal Top-60 retained in every selection",
            "rescue_sources": list(RESCUE_SOURCES),
            "rescue_budgets_per_match": list(RESCUE_BUDGETS),
            "temporal_gap_seconds": GAP_SECONDS,
            "delivery_window": "20+40 seconds from frozen candidate scores",
            "rescue_filter": "skip duplicate groups and candidates within 45s of any protected or previously added candidate",
            "goal_coverage": "catalog timestamp inside any selected candidate delivery interval",
            "cost": "unique merged video-minutes and merged segments; computed against the same protected Top-60 baseline",
        },
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]
        ],
        "interpretation": "Exploratory post-hoc additive rescue curve on the same TRAIN/LOMO population; not independent validation and not a promotion decision. No VALIDATION or TEST accessed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": args.output.name, "variants": len(summary_rows), "matches": len(by_match), "goals": sum(map(len, goals.values()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
