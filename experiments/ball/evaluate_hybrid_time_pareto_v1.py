"""TRAIN-only recall/time curve for frozen rankings and rescue policies."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_complementarity_matrix_v1 import (
    BASE_SCORES, EXPERTS, GAP_SECONDS, ROOT, RRF_OFFSET,
    merged_cost, read_csv, select_with_gap, sha256, write_csv,
)

DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/HYBRID-TIME-PARETO-EXP-0001"
WINDOWS = {30: (10.0, 20.0), 40: (40.0 / 3, 80.0 / 3), 45: (15.0, 30.0), 60: (20.0, 40.0)}
LIMITS = (20, 30, 40, 50, 55, 60)


def center(row: dict[str, object]) -> float:
    return float(row["delivery_start_seconds"]) + 20.0


def build_rrf(rows: list[dict[str, object]], sources: tuple[str, ...], name: str) -> list[dict[str, object]]:
    scores: dict[str, float] = defaultdict(float)
    for source in sources:
        ordered = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        for rank, row in enumerate(ordered, 1):
            scores[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
    return sorted(rows, key=lambda row: (-scores[str(row["group_uid"])], str(row["group_uid"])))


def rescue_order(core: list[dict[str, object]], sources: list[tuple[list[dict[str, object]], int]], target: int) -> list[dict[str, object]]:
    chosen = list(core[:target])
    if len(chosen) >= target:
        return chosen
    for source_rows, quota in sources:
        added = 0
        for row in source_rows:
            if len(chosen) >= target or added >= quota:
                break
            if any(str(row["group_uid"]) == str(other["group_uid"]) for other in chosen):
                continue
            if any(abs(center(row) - center(other)) < GAP_SECONDS for other in chosen):
                continue
            chosen.append(row)
            added += 1
        if len(chosen) >= target:
            break
    return chosen


def coverage(selected: dict[str, list[dict[str, object]]], goals: dict[str, list[dict[str, str]]]) -> tuple[int, list[dict[str, object]]]:
    total = 0
    rows: list[dict[str, object]] = []
    for match, candidates in selected.items():
        match_covered = 0
        for goal in goals[match]:
            timestamp = float(goal["event_timestamp_seconds"])
            hit = any(float(row["delivery_start_seconds"]) <= timestamp <= float(row["delivery_end_seconds"]) for row in candidates)
            match_covered += int(hit)
            rows.append({"match_id": match, "goal_catalog_id": goal["goal_catalog_id"], "covered": hit})
        total += match_covered
    return total, rows


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
    experts = {
        name: {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in read_csv(path)}
        for name, path in EXPERTS.items()
    }
    common = set(pool_by_key)
    for scores in experts.values():
        common &= set(scores)
    if len(common) != 5372:
        raise ValueError(f"Expected frozen common TRAIN population 5,372; found {len(common)}")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(common):
        row: dict[str, object] = dict(pool_by_key[key])
        for source, scores in experts.items():
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

    per_match_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for match, rows in by_match.items():
        # Fixed within-match rankings and rescue sequences are independent of window size.
        temporal = sorted(rows, key=lambda r: (-float(r["rank_score"]), str(r["group_uid"])))
        rrf3 = build_rrf(rows, BASE_SCORES, "rrf3")
        rrf4net = build_rrf(rows, (*BASE_SCORES, "net_motion"), "rrf4net")
        fusion = sorted(rows, key=lambda r: (-float(r["fusion_score_goal"]), str(r["group_uid"])))
        net = sorted(rows, key=lambda r: (-float(r["net_motion"]), str(r["group_uid"])))

        for window, (pre, post) in WINDOWS.items():
            # Rebuild temporal lists with this window, holding each candidate's center fixed.
            wrows = []
            for row in rows:
                c = center(row)
                start, end = max(0.0, c - pre), c + post
                wrows.append({**row, "delivery_start_seconds": start, "delivery_end_seconds": end})
            ordered = {
                "temporal": sorted(wrows, key=lambda r: (-float(r["rank_score"]), str(r["group_uid"]))),
                "RRF3": build_rrf(wrows, BASE_SCORES, "rrf3"),
                "RRF4_net": build_rrf(wrows, (*BASE_SCORES, "net_motion"), "rrf4net"),
            }
            # The two simple CORE+RESCUE policies measured in A2. Their flattened order
            # makes K comparable: temporal core first, followed by non-overlapping rescues.
            for k in LIMITS:
                policies: dict[str, list[dict[str, object]]] = {
                    "temporal": select_with_gap(ordered["temporal"], k),
                    "RRF3": select_with_gap(ordered["RRF3"], k),
                    "RRF4_net": select_with_gap(ordered["RRF4_net"], k),
                }
                temporal_core = select_with_gap(ordered["temporal"], 35)
                fusion_w = sorted(wrows, key=lambda r: (-float(r["fusion_score_goal"]), str(r["group_uid"])))
                net_w = sorted(wrows, key=lambda r: (-float(r["net_motion"]), str(r["group_uid"])))
                policies["CORE35_FUSION10_NET10"] = rescue_order(temporal_core, [(fusion_w, 10), (net_w, 10)], k)
                temporal_core40 = select_with_gap(ordered["temporal"], 40)
                policies["CORE40_RRF4_NET_RESCUE"] = rescue_order(temporal_core40, [(ordered["RRF4_net"], 20)], k)

                for method, chosen in policies.items():
                    selected = {match: chosen}
                    covered, goal_details = coverage(selected, {match: goals[match]})
                    segments, minutes = merged_cost(selected)
                    per_match_rows.append({
                        "method": method, "window_seconds": window, "k_per_match": k,
                        "match_id": match, "catalog_goals": len(goals[match]),
                        "covered_goals": covered, "selected_candidates": len(chosen),
                        "merged_segments": segments, "unique_review_minutes": f"{minutes:.3f}",
                    })
                    for row in chosen:
                        candidate_rows.append({
                            "method": method, "window_seconds": window, "k_per_match": k,
                            "match_id": match, "group_uid": row["group_uid"],
                            "delivery_start_seconds": f"{float(row['delivery_start_seconds']):.3f}",
                            "delivery_end_seconds": f"{float(row['delivery_end_seconds']):.3f}",
                        })

    aggregates: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_match_rows:
        aggregates[(row["method"], row["window_seconds"], row["k_per_match"])].append(row)
    summaries: list[dict[str, object]] = []
    for (method, window, k), group in sorted(aggregates.items()):
        total_goals = sum(int(row["catalog_goals"]) for row in group)
        covered_goals = sum(int(row["covered_goals"]) for row in group)
        summaries.append({
            "method": method, "window_seconds": int(window), "k_per_match": int(k),
            "catalog_goals": total_goals, "covered_goals": covered_goals,
            "recall": f"{covered_goals / total_goals:.6f}",
            "selected_candidates": sum(int(row["selected_candidates"]) for row in group),
            "merged_segments": sum(int(row["merged_segments"]) for row in group),
            "unique_review_minutes_per_match": f"{sum(float(row['unique_review_minutes']) for row in group) / len(group):.3f}",
            "matches_with_incremental_coverage_vs_temporal": "",
        })
    # Add distribution information relative to the temporal method at the same K/window.
    temporal_keys = {(r["window_seconds"], r["k_per_match"]): r for r in summaries if r["method"] == "temporal"}
    for summary_row in summaries:
        key = (summary_row["window_seconds"], summary_row["k_per_match"])
        if summary_row["method"] == "temporal":
            summary_row["matches_with_incremental_coverage_vs_temporal"] = 0
            continue
        method_rows = [r for r in per_match_rows if r["method"] == summary_row["method"] and int(r["window_seconds"]) == key[0] and int(r["k_per_match"]) == key[1]]
        baseline_rows = {r["match_id"]: int(r["covered_goals"]) for r in per_match_rows if r["method"] == "temporal" and int(r["window_seconds"]) == key[0] and int(r["k_per_match"]) == key[1]}
        match_gains = sum(int(r["covered_goals"]) > baseline_rows[r["match_id"]] for r in method_rows)
        summary_row["matches_with_incremental_coverage_vs_temporal"] = match_gains

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "hybrid_time_pareto_summary_v1.csv", summaries)
    write_csv(args.output / "hybrid_time_pareto_by_match_v1.csv", per_match_rows)
    write_csv(args.output / "hybrid_time_pareto_selected_v1.csv", candidate_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matches; 5,372 common groups; 130 completed goals; no VALIDATION or TEST",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]],
        "windows": {str(k): {"pre_seconds": v[0], "post_seconds": v[1]} for k, v in WINDOWS.items()},
        "k_values": list(LIMITS), "gap_seconds": GAP_SECONDS,
        "methods": ["temporal", "RRF3", "RRF4_net", "CORE35_FUSION10_NET10", "CORE40_RRF4_NET_RESCUE"],
        "interpretation": "Exploratory TRAIN/LOMO curve; post-hoc selection; not independent validation or promotion. The rescue variants retain the temporal core first.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(summaries), "per_match": len(per_match_rows), "matches": len(by_match), "goals": sum(map(len, goals.values()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
