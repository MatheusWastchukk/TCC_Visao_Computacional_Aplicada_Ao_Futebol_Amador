"""Diagnostic LOMO-style adaptive review budget using an unlabeled RRF4 score-gap feature."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_complementarity_matrix_v1 import EXPERTS, ROOT, read_csv, sha256, write_csv
from evaluate_review_budget_selection_v1 import measure, rrf_score, select_rank_budget

DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/ADAPTIVE-MATCH-BUDGET-EXP-0001"
FIXED_BUDGET = 45
BUDGET_BY_TIER = {"EASY": 30, "MEDIUM": 45, "HARD": 60}


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")

    pool = read_csv(args.candidate_scores)
    base = {(row["match_id"], row["group_uid"]): row for row in pool}
    expert_maps = {
        name: {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in read_csv(path)}
        for name, path in EXPERTS.items()
    }
    common = set(base)
    for values in expert_maps.values():
        common &= set(values)
    if len(common) != 5372:
        raise ValueError(f"Expected 5,372 common TRAIN groups; found {len(common)}")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for match_id, group_uid in sorted(common):
        row: dict[str, object] = dict(base[(match_id, group_uid)])
        for source in EXPERTS:
            row[source] = expert_maps[source][(match_id, group_uid)]
        by_match[match_id].append(row)
    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Only TRAIN goals are allowed")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or set(by_match) != set(goals) or sum(map(len, goals.values())) != 130:
        raise ValueError("Expected 22 TRAIN matches and 130 completed goals")

    features: dict[str, float] = {}
    ranked_by_match: dict[str, list[dict[str, object]]] = {}
    for match_id, raw in by_match.items():
        scores = rrf_score(raw)
        ranked = []
        for row in raw:
            item = dict(row)
            item["rrf4_net_score"] = scores[str(item["group_uid"])]
            ranked.append(item)
        ranked.sort(key=lambda item: (-float(item["rrf4_net_score"]), str(item["group_uid"])))
        ranked_by_match[match_id] = ranked
        # Unlabeled separability feature fixed before evaluation: mean RRF4 score
        # of ranks 1-10 minus the score at rank 50.
        features[match_id] = sum(float(row["rrf4_net_score"]) for row in ranked[:10]) / 10.0 - float(ranked[49]["rrf4_net_score"])

    match_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for heldout in sorted(by_match):
        development = [features[match] for match in features if match != heldout]
        q33, q67 = quantile(development, 1/3), quantile(development, 2/3)
        feature = features[heldout]
        tier = "EASY" if feature >= q67 else "MEDIUM" if feature >= q33 else "HARD"
        budget = BUDGET_BY_TIER[tier]
        thresholds = {"heldout_match": heldout, "feature_name": "mean_rrf4_rank1_10_minus_rank50", "feature_value": f"{feature:.10f}", "p33_development": f"{q33:.10f}", "p67_development": f"{q67:.10f}", "tier": tier, "assigned_budget_minutes": budget}
        threshold_rows.append(thresholds)
        selections = {
            "fixed_45_min": select_rank_budget(ranked_by_match[heldout], "rrf4_net_score", FIXED_BUDGET*60),
            "adaptive": select_rank_budget(ranked_by_match[heldout], "rrf4_net_score", budget*60),
        }
        measured: dict[str, tuple[int, float, int, set[str]]] = {}
        for method, chosen in selections.items():
            covered, minutes, segments = measure(chosen, goals[heldout])
            ids = {
                goal["goal_catalog_id"] for goal in goals[heldout]
                if any(float(row["delivery_start_seconds"]) <= float(goal["event_timestamp_seconds"]) <= float(row["delivery_end_seconds"]) for row in chosen)
            }
            measured[method] = (covered, minutes, segments, ids)
            for rank, row in enumerate(chosen, 1):
                selected_rows.append({
                    "method": method, "heldout_match": heldout, "group_uid": row["group_uid"],
                    "rank": rank, "feature_value": f"{feature:.10f}", "tier": tier,
                    "assigned_budget_minutes": budget,
                    "delivery_start_seconds": row["delivery_start_seconds"],
                    "delivery_end_seconds": row["delivery_end_seconds"],
                })
        base_result, adaptive = measured["fixed_45_min"], measured["adaptive"]
        match_rows.append({
            **thresholds,
            "catalog_goals": len(goals[heldout]),
            "fixed_45_covered": base_result[0], "adaptive_covered": adaptive[0],
            "gained_vs_fixed": len(adaptive[3]-base_result[3]), "lost_vs_fixed": len(base_result[3]-adaptive[3]),
            "fixed_45_unique_minutes": f"{base_result[1]:.3f}", "adaptive_unique_minutes": f"{adaptive[1]:.3f}",
            "fixed_45_segments": base_result[2], "adaptive_segments": adaptive[2],
            "fixed_45_candidates": len(selections["fixed_45_min"]), "adaptive_candidates": len(selections["adaptive"]),
        })

    total_goals = sum(int(row["catalog_goals"]) for row in match_rows)
    summary = [{
        "comparison": "adaptive_vs_fixed_45_min_RRF4_rank",
        "lomo_matches": len(match_rows), "catalog_goals": total_goals,
        "fixed_45_covered": sum(int(row["fixed_45_covered"]) for row in match_rows),
        "adaptive_covered": sum(int(row["adaptive_covered"]) for row in match_rows),
        "goals_gained": sum(int(row["gained_vs_fixed"]) for row in match_rows),
        "goals_lost": sum(int(row["lost_vs_fixed"]) for row in match_rows),
        "mean_fixed_45_minutes": f"{sum(float(row['fixed_45_unique_minutes']) for row in match_rows)/len(match_rows):.3f}",
        "mean_adaptive_minutes": f"{sum(float(row['adaptive_unique_minutes']) for row in match_rows)/len(match_rows):.3f}",
        "fixed_45_segments": sum(int(row["fixed_45_segments"]) for row in match_rows),
        "adaptive_segments": sum(int(row["adaptive_segments"]) for row in match_rows),
        "mean_assigned_budget": f"{sum(int(row['assigned_budget_minutes']) for row in match_rows)/len(match_rows):.3f}",
        "easy_matches": sum(row["tier"] == "EASY" for row in match_rows),
        "medium_matches": sum(row["tier"] == "MEDIUM" for row in match_rows),
        "hard_matches": sum(row["tier"] == "HARD" for row in match_rows),
    }]

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "adaptive_match_budget_summary_v1.csv", summary)
    write_csv(args.output / "adaptive_match_budget_by_match_v1.csv", match_rows)
    write_csv(args.output / "adaptive_match_budget_thresholds_v1.csv", threshold_rows)
    write_csv(args.output / "adaptive_match_budget_selected_v1.csv", selected_rows)
    manifest = {
        "experiment_id": args.output.name,
        "scope": "TRAIN-only diagnostic LOMO; 22 matches; 5,372 common groups; 130 goals; no VALIDATION/TEST or V4 promotion",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]],
        "feature": "mean RRF4 score ranks 1-10 minus rank-50 score; candidate scores only, no match labels",
        "thresholds": "For each heldout match, P33/P67 of the feature across the other 21 matches; high gap=EASY 30min, middle=MEDIUM 45min, low=HARD 60min",
        "selection": "RRF4 rank with incremental union-of-intervals cost, matching the B2 ranked budget selector",
        "methodological_limit": "The frozen OOF scores for development matches may have been produced by models trained on the heldout match. This is not leakage-free nested LOMO; treat as diagnostic only.",
        "interpretation": "Exploratory TRAIN comparison against fixed 45 unique minutes per match; no promotion.",
    }
    (args.output / "manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matches": len(match_rows), "goals": total_goals, "summary": summary[0]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
