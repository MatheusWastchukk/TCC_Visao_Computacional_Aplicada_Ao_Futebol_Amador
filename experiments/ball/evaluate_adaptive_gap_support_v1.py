"""TRAIN-only test of a source-disagreement exception to the frozen 45s gap."""
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
DEFAULT_B1 = ROOT / "experiments/ball/HYBRID-TIME-PARETO-EXP-0001/hybrid_time_pareto_selected_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/ADAPTIVE-GAP-EXP-0001"
K = 60
SOURCES = (*BASE_SCORES, "net_motion")


def anchor(row: dict[str, object]) -> float:
    return float(row["delivery_start_seconds"]) + 20.0


def rrf_rank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scores: dict[str, float] = defaultdict(float)
    for source in SOURCES:
        ordered = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        for rank, row in enumerate(ordered, 1):
            scores[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
    return sorted(rows, key=lambda row: (-scores[str(row["group_uid"])] , str(row["group_uid"])))


def select_with_gap(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    chosen = []
    for row in rows:
        if all(abs(anchor(row) - anchor(other)) >= GAP_SECONDS for other in chosen):
            chosen.append(row)
            if len(chosen) >= limit:
                break
    return chosen


def baseline_core_rescue(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    temporal = sorted(rows, key=lambda row: (-float(row["rank_score"]), str(row["group_uid"])))
    core = select_with_gap(temporal, 40)
    rrf = rrf_rank(rows)
    chosen = list(core)
    added = 0
    for row in rrf:
        if len(chosen) >= K or added >= 20:
            break
        if any(str(row["group_uid"]) == str(other["group_uid"]) for other in chosen):
            continue
        if any(abs(anchor(row) - anchor(other)) < GAP_SECONDS for other in chosen):
            continue
        chosen.append(row)
        added += 1
    return chosen


def adaptive_core_rescue(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    rank_maps: dict[str, dict[str, int]] = {}
    for source in SOURCES:
        ordered = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        rank_maps[source] = {str(row["group_uid"]): rank for rank, row in enumerate(ordered, 1)}
    for row in rows:
        uid = str(row["group_uid"])
        row["support_sources"] = {source for source in SOURCES if rank_maps[source][uid] <= K}
        row["support_count"] = len(row["support_sources"])
    temporal = sorted(rows, key=lambda row: (-float(row["rank_score"]), str(row["group_uid"])))
    core = select_with_gap(temporal, 40)
    chosen = list(core)
    rrf = rrf_rank(rows)
    added = 0
    for row in rrf:
        if len(chosen) >= K or added >= 20:
            break
        uid = str(row["group_uid"])
        if any(uid == str(other["group_uid"]) for other in chosen):
            continue
        near = [other for other in chosen if abs(anchor(row) - anchor(other)) < GAP_SECONDS]
        if not near:
            chosen.append(row)
            added += 1
            continue
        # Permit a close CORE+RESCUE pair only when the candidate has support in
        # at least two methods and its source support is disjoint from every
        # nearby selected candidate. Otherwise retain the existing higher-ranked item.
        support = row["support_sources"]
        if len(support) >= 2 and all(not (support & other.get("support_sources", set())) for other in near):
            chosen.append(row)
            added += 1
    return chosen, rank_maps


def to_60s(row: dict[str, object]) -> dict[str, object]:
    center = anchor(row)
    return {**row, "delivery_start_seconds": max(0.0, center-20.0), "delivery_end_seconds": center+40.0}


def measure_goals(rows: list[dict[str, object]], goals: list[dict[str, str]]) -> tuple[set[str], int, float]:
    covered = {
        goal["goal_catalog_id"] for goal in goals
        if any(float(row["delivery_start_seconds"]) <= float(goal["event_timestamp_seconds"]) <= float(row["delivery_end_seconds"]) for row in rows)
    }
    segments, minutes = merged_cost({"one_match": rows})
    return covered, segments, minutes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--b1-selected", type=Path, default=DEFAULT_B1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")
    pool = read_csv(args.candidate_scores)
    pool_by_key = {(row["match_id"], row["group_uid"]): row for row in pool}
    experts = {
        name: {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in read_csv(path)}
        for name, path in EXPERTS.items()
    }
    common = set(pool_by_key)
    for score_rows in experts.values():
        common &= set(score_rows)
    if len(common) != 5372:
        raise ValueError(f"Expected 5,372 common TRAIN groups; got {len(common)}")
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(common):
        row: dict[str, object] = dict(pool_by_key[key])
        for source in EXPERTS:
            row[source] = experts[source][key]
        by_match[key[0]].append(row)
    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Only TRAIN goal labels are allowed")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or sum(map(len, goals.values())) != 130 or set(by_match) != set(goals):
        raise ValueError("Expected 22 TRAIN matches and 130 goals")

    b1_rows = [row for row in read_csv(args.b1_selected) if row["method"] == "CORE40_RRF4_NET_RESCUE" and row["window_seconds"] == "60" and row["k_per_match"] == "60"]
    b1_order: dict[str, list[str]] = defaultdict(list)
    for row in b1_rows:
        b1_order[row["match_id"]].append(row["group_uid"])

    summary_rows: list[dict[str, object]] = []
    by_match_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    total_goal_gains = total_goal_losses = 0
    for match_id, raw in by_match.items():
        baseline = baseline_core_rescue(raw)
        expected = b1_order.get(match_id, [])
        if [str(row["group_uid"]) for row in baseline] != expected:
            raise ValueError(f"B1 baseline selection failed parity check in {match_id}")
        adaptive, rank_maps = adaptive_core_rescue(raw)
        base_delivery = [to_60s(row) for row in baseline]
        adaptive_delivery = [to_60s(row) for row in adaptive]
        base_hits, base_segments, base_minutes = measure_goals(base_delivery, goals[match_id])
        adapt_hits, adapt_segments, adapt_minutes = measure_goals(adaptive_delivery, goals[match_id])
        gains, losses = len(adapt_hits-base_hits), len(base_hits-adapt_hits)
        total_goal_gains += gains
        total_goal_losses += losses
        row = {
            "match_id": match_id, "catalog_goals": len(goals[match_id]),
            "baseline_candidates": len(baseline), "adaptive_candidates": len(adaptive),
            "baseline_covered": len(base_hits), "adaptive_covered": len(adapt_hits),
            "gained_vs_gap45": gains, "lost_vs_gap45": losses,
            "baseline_minutes": f"{base_minutes:.3f}", "adaptive_minutes": f"{adapt_minutes:.3f}",
            "baseline_segments": base_segments, "adaptive_segments": adapt_segments,
            "adaptive_close_pairs_kept": sum(
                abs(anchor(adaptive[i])-anchor(adaptive[j])) < GAP_SECONDS
                for i in range(len(adaptive)) for j in range(i+1, len(adaptive))
            ),
        }
        by_match_rows.append(row)
        for variant, chosen in (("gap45_baseline", baseline), ("adaptive_support_gap", adaptive)):
            for rank, item in enumerate(chosen, 1):
                support = item.get("support_sources", set())
                selected_rows.append({
                    "variant": variant, "match_id": match_id, "rank": rank, "group_uid": item["group_uid"],
                    "anchor_seconds": f"{anchor(item):.3f}", "support_count_top60": len(support),
                    "support_sources": ",".join(sorted(support)),
                    "temporal_rank": rank_maps["rank_score"].get(str(item["group_uid"]), ""),
                    "visual_rank": rank_maps["visual_score_goal"].get(str(item["group_uid"]), ""),
                    "fusion_rank": rank_maps["fusion_score_goal"].get(str(item["group_uid"]), ""),
                    "net_rank": rank_maps["net_motion"].get(str(item["group_uid"]), ""),
                })

    total_goals = sum(int(row["catalog_goals"]) for row in by_match_rows)
    summary_rows.append({
        "variant": "gap45_baseline", "matches": len(by_match_rows), "catalog_goals": total_goals,
        "covered_goals": sum(int(row["baseline_covered"]) for row in by_match_rows),
        "selected_candidates": sum(int(row["baseline_candidates"]) for row in by_match_rows),
        "mean_unique_minutes_per_match": f"{sum(float(row['baseline_minutes']) for row in by_match_rows)/len(by_match_rows):.3f}",
        "total_segments": sum(int(row["baseline_segments"]) for row in by_match_rows),
        "close_pairs_kept": 0, "gained_vs_baseline": 0, "lost_vs_baseline": 0,
    })
    summary_rows.append({
        "variant": "adaptive_support_gap", "matches": len(by_match_rows), "catalog_goals": total_goals,
        "covered_goals": sum(int(row["adaptive_covered"]) for row in by_match_rows),
        "selected_candidates": sum(int(row["adaptive_candidates"]) for row in by_match_rows),
        "mean_unique_minutes_per_match": f"{sum(float(row['adaptive_minutes']) for row in by_match_rows)/len(by_match_rows):.3f}",
        "total_segments": sum(int(row["adaptive_segments"]) for row in by_match_rows),
        "close_pairs_kept": sum(int(row["adaptive_close_pairs_kept"]) for row in by_match_rows),
        "gained_vs_baseline": total_goal_gains, "lost_vs_baseline": total_goal_losses,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "adaptive_gap_summary_v1.csv", summary_rows)
    write_csv(args.output / "adaptive_gap_by_match_v1.csv", by_match_rows)
    write_csv(args.output / "adaptive_gap_selected_v1.csv", selected_rows)
    manifest = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matches; 5,372 common groups; 130 goals; no VALIDATION/TEST or V4 promotion",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values(), args.b1_selected]],
        "baseline": "Parity-checked exact B1 CORE40_RRF4_NET_RESCUE, 60s, K60, gap45 selected IDs and order.",
        "adaptive_rule": "Start with temporal CORE40 gap45. Add in RRF4 order. A candidate within 45s of any selected item is kept only if it supports >=2 of the four RRF4 sources in their Top-60 and its support-source set is disjoint from every nearby selected item's set. Otherwise reject as redundant/weak. Cap K=60.",
        "support_features": "Frozen LOMO rank scores only; no goal labels are used in rule decisions.",
        "interpretation": "Exploratory post-hoc TRAIN test; label scores not used for selection. No V4 promotion.",
    }
    (args.output / "manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matches": len(by_match_rows), "goals": total_goals, "summary": summary_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
