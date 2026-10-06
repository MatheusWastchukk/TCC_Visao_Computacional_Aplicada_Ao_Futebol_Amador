"""TRAIN-only comparison of per-support-tier delivery windows at fixed RRF4 ranks."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_complementarity_matrix_v1 import (
    BASE_SCORES, EXPERTS, GAP_SECONDS, ROOT, RRF_OFFSET,
    merged_cost, read_csv, sha256, write_csv,
)

DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/TIERED-WINDOW-EXP-0001"
K_VALUES = (20, 30, 40, 50, 55, 60)
WINDOWS = {
    "uniform_60_60_60": {"A": 60, "B": 60, "C": 60},
    "tier_30_40_60": {"A": 30, "B": 40, "C": 60},
    "tier_35_45_60": {"A": 35, "B": 45, "C": 60},
    "tier_40_45_60": {"A": 40, "B": 45, "C": 60},
}


def rrf4_rank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scores: dict[str, float] = defaultdict(float)
    sources = (*BASE_SCORES, "net_motion")
    for source in sources:
        ordered = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        for rank, row in enumerate(ordered, 1):
            scores[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
    return sorted(rows, key=lambda row: (-scores[str(row["group_uid"])], str(row["group_uid"])))


def select_gap(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        center = float(row["delivery_start_seconds"]) + 20.0
        if all(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) >= GAP_SECONDS for other in selected):
            selected.append(row)
            if len(selected) == limit:
                break
    return selected


def goal_hits(rows: list[dict[str, object]], goals: list[dict[str, str]]) -> set[str]:
    hits = set()
    for goal in goals:
        timestamp = float(goal["event_timestamp_seconds"])
        if any(float(row["delivery_start_seconds"]) <= timestamp <= float(row["delivery_end_seconds"]) for row in rows):
            hits.add(goal["goal_catalog_id"])
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")

    pool = read_csv(args.candidate_scores)
    by_key = {(row["match_id"], row["group_uid"]): row for row in pool}
    experts = {
        name: {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in read_csv(path)}
        for name, path in EXPERTS.items()
    }
    common = set(by_key)
    for source_scores in experts.values():
        common &= set(source_scores)
    if len(common) != 5372:
        raise ValueError(f"Expected 5,372 common TRAIN groups; found {len(common)}")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    score_sources = (*BASE_SCORES, "net_motion")
    for match_id, group_uid in sorted(common):
        row: dict[str, object] = dict(by_key[(match_id, group_uid)])
        row["net_motion"] = experts["net_motion"][(match_id, group_uid)]
        row["_group_key"] = f"{match_id}|{group_uid}"
        by_match[match_id].append(row)

    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Only TRAIN labels may be scored")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or set(by_match) != set(goals) or sum(map(len, goals.values())) != 130:
        raise ValueError("Expected the frozen population of 22 matches and 130 TRAIN goals")

    summary: list[dict[str, object]] = []
    by_match_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for match_id, raw in by_match.items():
        rank_by_source: dict[str, dict[str, int]] = {}
        for source in score_sources:
            order = sorted(raw, key=lambda row: (-float(row[source]), str(row["group_uid"])))
            rank_by_source[source] = {str(row["group_uid"]): rank for rank, row in enumerate(order, 1)}
        ranked = rrf4_rank(raw)
        for row in ranked:
            uid = str(row["group_uid"])
            support = sum(rank_by_source[source][uid] <= 60 for source in score_sources)
            row["support_count_top60"] = support
            row["temporal_rank"] = rank_by_source["rank_score"][uid]
            row["tier"] = "A" if support >= 3 or row["temporal_rank"] <= 35 else ("B" if support == 2 else "C")

        for k in K_VALUES:
            chosen = select_gap(ranked, k)
            by_variant_hits: dict[str, set[str]] = {}
            for variant, spec in WINDOWS.items():
                delivered: list[dict[str, object]] = []
                tiers = {tier: 0 for tier in "ABC"}
                for row in chosen:
                    tier = str(row["tier"])
                    window = spec[tier]
                    # Match B1's centered delivery convention: the candidate anchor
                    # is original delivery_start_seconds + 20 seconds.
                    anchor = float(row["delivery_start_seconds"]) + 20.0
                    pre = window / 3.0
                    post = window - pre
                    delivered.append({
                        **row,
                        "delivery_start_seconds": max(0.0, anchor - pre),
                        "delivery_end_seconds": anchor + post,
                        "assigned_tier": tier,
                        "assigned_window_seconds": window,
                    })
                    tiers[tier] += 1
                hits = goal_hits(delivered, goals[match_id])
                by_variant_hits[variant] = hits
                segments, minutes = merged_cost({match_id: delivered})
                by_match_rows.append({
                    "variant": variant, "k_per_match": k, "match_id": match_id,
                    "catalog_goals": len(goals[match_id]), "covered_goals": len(hits),
                    "selected_candidates": len(delivered), "tier_a_count": tiers["A"],
                    "tier_b_count": tiers["B"], "tier_c_count": tiers["C"],
                    "merged_segments": segments, "unique_review_minutes": f"{minutes:.3f}",
                })
                for rank, row in enumerate(delivered, 1):
                    selected_rows.append({
                        "variant": variant, "k_per_match": k, "match_id": match_id,
                        "group_uid": row["group_uid"], "selection_rank_rrf4": rank,
                        "tier": row["assigned_tier"], "support_count_top60": row["support_count_top60"],
                        "temporal_rank": row["temporal_rank"], "window_seconds": row["assigned_window_seconds"],
                        "delivery_start_seconds": f"{float(row['delivery_start_seconds']):.3f}",
                        "delivery_end_seconds": f"{float(row['delivery_end_seconds']):.3f}",
                    })

            base_hits = by_variant_hits["uniform_60_60_60"]
            for variant, hits in by_variant_hits.items():
                row = next(item for item in reversed(by_match_rows) if item["variant"] == variant and item["k_per_match"] == k and item["match_id"] == match_id)
                row["gained_vs_uniform"] = len(hits - base_hits)
                row["lost_vs_uniform"] = len(base_hits - hits)

    aggregates: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in by_match_rows:
        aggregates[(str(row["variant"]), int(row["k_per_match"]))].append(row)
    for (variant, k), rows in sorted(aggregates.items()):
        total_goals = sum(int(row["catalog_goals"]) for row in rows)
        summary.append({
            "variant": variant, "k_per_match": k, "catalog_goals": total_goals,
            "covered_goals": sum(int(row["covered_goals"]) for row in rows),
            "recall": f"{sum(int(row['covered_goals']) for row in rows) / total_goals:.6f}",
            "selected_candidates": sum(int(row["selected_candidates"]) for row in rows),
            "mean_unique_review_minutes_per_match": f"{sum(float(row['unique_review_minutes']) for row in rows) / len(rows):.3f}",
            "total_segments": sum(int(row["merged_segments"]) for row in rows),
            "matches_with_gains_vs_uniform": sum(int(row["gained_vs_uniform"]) > int(row["lost_vs_uniform"]) for row in rows),
            "matches_with_losses_vs_uniform": sum(int(row["lost_vs_uniform"]) > int(row["gained_vs_uniform"]) for row in rows),
            "goals_gained_vs_uniform": sum(int(row["gained_vs_uniform"]) for row in rows),
            "goals_lost_vs_uniform": sum(int(row["lost_vs_uniform"]) for row in rows),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "tiered_window_summary_v1.csv", summary)
    write_csv(args.output / "tiered_window_by_match_v1.csv", by_match_rows)
    write_csv(args.output / "tiered_window_selected_v1.csv", selected_rows)
    manifest = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matches; 5,372 common groups; 130 goals; no VALIDATION/TEST or V4 promotion",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]],
        "selection": "RRF4 temporal+visual+fusion+net_motion; fixed ranking and 45s gap; K in 20/30/40/50/55/60; tiers use only scores, never goal labels",
        "tier_rule": "A=support in >=3 of the four RRF4 score sources within each source Top-60 OR temporal rank<=35; B=exactly 2 source Top-60 supports; C=remaining selected candidates (single/zero Top-60 support, rescue-like)",
        "window_convention": "Anchor=original delivery_start_seconds+20; pre=one-third of assigned window, post=two-thirds; same convention as B1",
        "variants": WINDOWS,
        "k_values": list(K_VALUES), "gap_seconds": GAP_SECONDS,
        "interpretation": "Exploratory post-hoc comparison on the shared TRAIN/LOMO population; any gain requires independent confirmation.",
    }
    (args.output / "manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(summary), "per_match_rows": len(by_match_rows), "selected_rows": len(selected_rows), "matches": len(by_match), "goals": 130}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
