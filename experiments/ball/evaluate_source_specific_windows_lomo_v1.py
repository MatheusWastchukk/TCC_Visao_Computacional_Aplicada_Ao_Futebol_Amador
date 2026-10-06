"""LOMO evaluation of source-specific asymmetric windows, fitted on other TRAIN matches."""
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
DEFAULT_OUTPUT = ROOT / "experiments/ball/SOURCE-WINDOW-LOMO-EXP-0001"
K = 60
Q_LOW, Q_HIGH = 0.05, 0.95
SOURCES = ("temporal", "fusion", "net_motion", "RRF4", "rescue")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot fit quantile without development offsets")
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def rrf4_rank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scores: dict[str, float] = defaultdict(float)
    for source in (*BASE_SCORES, "net_motion"):
        ordered = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        for rank, row in enumerate(ordered, 1):
            scores[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
    return sorted(rows, key=lambda row: (-scores[str(row["group_uid"])] , str(row["group_uid"])))


def select_gap(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        anchor = float(row["delivery_start_seconds"]) + 20.0
        if all(abs(anchor - (float(other["delivery_start_seconds"]) + 20.0)) >= GAP_SECONDS for other in selected):
            selected.append(row)
            if len(selected) == limit:
                break
    return selected


def rescue_order(core: list[dict[str, object]], source_rows: list[tuple[list[dict[str, object]], int]]) -> list[dict[str, object]]:
    chosen = list(core[:K])
    for rows, quota in source_rows:
        added = 0
        for row in rows:
            if len(chosen) >= K or added >= quota:
                break
            uid = str(row["group_uid"])
            if any(str(other["group_uid"]) == uid for other in chosen):
                continue
            anchor = float(row["delivery_start_seconds"]) + 20.0
            if any(abs(anchor - (float(other["delivery_start_seconds"]) + 20.0)) < GAP_SECONDS for other in chosen):
                continue
            chosen.append(row)
            added += 1
    return chosen


def candidate_lists(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    temporal = sorted(rows, key=lambda row: (-float(row["rank_score"]), str(row["group_uid"])))
    fusion = sorted(rows, key=lambda row: (-float(row["fusion_score_goal"]), str(row["group_uid"])))
    net = sorted(rows, key=lambda row: (-float(row["net_motion"]), str(row["group_uid"])))
    rrf = rrf4_rank(rows)
    core = select_gap(temporal, 35)
    return {
        "temporal": select_gap(temporal, K),
        "fusion": select_gap(fusion, K),
        "net_motion": select_gap(net, K),
        "RRF4": select_gap(rrf, K),
        "rescue": rescue_order(core, [(fusion, 10), (net, 10)]),
    }


def anchor(row: dict[str, object]) -> float:
    return float(row["delivery_start_seconds"]) + 20.0


def base_interval(row: dict[str, object]) -> tuple[float, float]:
    center = anchor(row)
    return max(0.0, center - 20.0), center + 40.0


def goal_hits(rows: list[dict[str, object]], goals: list[dict[str, str]], windows: tuple[float, float] | None) -> set[str]:
    hits: set[str] = set()
    for goal in goals:
        timestamp = float(goal["event_timestamp_seconds"])
        for row in rows:
            if windows is None:
                start, end = base_interval(row)
            else:
                low, high = windows
                center = anchor(row)
                start, end = max(0.0, center + low), max(0.0, center + high)
            if start <= timestamp <= end:
                hits.add(goal["goal_catalog_id"])
                break
    return hits


def nearest_covering_delta(rows: list[dict[str, object]], goals: list[dict[str, str]]) -> list[dict[str, object]]:
    offsets = []
    for goal in goals:
        timestamp = float(goal["event_timestamp_seconds"])
        eligible = [row for row in rows if base_interval(row)[0] <= timestamp <= base_interval(row)[1]]
        if eligible:
            chosen = min(eligible, key=lambda row: (abs(timestamp - anchor(row)), str(row["group_uid"])))
            offsets.append({
                "goal_catalog_id": goal["goal_catalog_id"],
                "group_uid": chosen["group_uid"],
                "candidate_anchor_seconds": f"{anchor(chosen):.3f}",
                "goal_timestamp_seconds": f"{timestamp:.3f}",
                "delta_goal_minus_candidate_seconds": f"{timestamp - anchor(chosen):.3f}",
            })
    return offsets


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
        raise ValueError(f"Expected the frozen 5,372 common TRAIN groups; found {len(common)}")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for match_id, group_uid in sorted(common):
        row: dict[str, object] = dict(pool_by_key[(match_id, group_uid)])
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

    selections = {match: candidate_lists(rows) for match, rows in by_match.items()}
    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    offset_rows: list[dict[str, object]] = []
    for source in SOURCES:
        for heldout in sorted(by_match):
            development_offsets: list[float] = []
            training_goals = 0
            for dev_match in sorted(by_match):
                if dev_match == heldout:
                    continue
                offsets = nearest_covering_delta(selections[dev_match][source], goals[dev_match])
                training_goals += len(offsets)
                for row in offsets:
                    development_offsets.append(float(row["delta_goal_minus_candidate_seconds"]))
                    offset_rows.append({
                        "source": source, "heldout_match": heldout, "development_match": dev_match,
                        **row,
                    })
            if not development_offsets:
                raise ValueError(f"No development offsets for {source}, heldout={heldout}")
            low, high = quantile(development_offsets, Q_LOW), quantile(development_offsets, Q_HIGH)
            if low >= high:
                raise ValueError(f"Invalid learned interval for {source}/{heldout}: {low}..{high}")
            selected = selections[heldout][source]
            baseline = goal_hits(selected, goals[heldout], None)
            learned = goal_hits(selected, goals[heldout], (low, high))
            base_intervals = []
            learned_intervals = []
            for row in selected:
                start, end = base_interval(row)
                base_intervals.append({**row, "delivery_start_seconds": start, "delivery_end_seconds": end})
                center = anchor(row)
                learned_intervals.append({
                    **row,
                    "delivery_start_seconds": max(0.0, center + low),
                    "delivery_end_seconds": max(0.0, center + high),
                })
            base_segments, base_minutes = merged_cost({heldout: base_intervals})
            learned_segments, learned_minutes = merged_cost({heldout: learned_intervals})
            record = {
                "source": source, "heldout_match": heldout, "development_offset_count": len(development_offsets),
                "development_goals_covered": training_goals, "p05_delta_seconds": f"{low:.3f}",
                "p95_delta_seconds": f"{high:.3f}", "learned_window_seconds": f"{high-low:.3f}",
                "heldout_goals": len(goals[heldout]), "baseline_60s_covered": len(baseline),
                "learned_covered": len(learned), "goals_gained_vs_60s": len(learned-baseline),
                "goals_lost_vs_60s": len(baseline-learned), "selected_candidates": len(selected),
                "baseline_unique_minutes": f"{base_minutes:.3f}", "learned_unique_minutes": f"{learned_minutes:.3f}",
                "baseline_segments": base_segments, "learned_segments": learned_segments,
            }
            match_rows.append(record)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in match_rows:
        grouped[str(row["source"])].append(row)
    for source, rows in grouped.items():
        total = sum(int(row["heldout_goals"]) for row in rows)
        summary_rows.append({
            "source": source, "lomo_matches": len(rows), "catalog_goals": total,
            "baseline_60s_covered": sum(int(row["baseline_60s_covered"]) for row in rows),
            "learned_covered": sum(int(row["learned_covered"]) for row in rows),
            "goals_gained_vs_60s": sum(int(row["goals_gained_vs_60s"]) for row in rows),
            "goals_lost_vs_60s": sum(int(row["goals_lost_vs_60s"]) for row in rows),
            "baseline_mean_minutes_per_match": f"{sum(float(row['baseline_unique_minutes']) for row in rows)/len(rows):.3f}",
            "learned_mean_minutes_per_match": f"{sum(float(row['learned_unique_minutes']) for row in rows)/len(rows):.3f}",
            "baseline_segments": sum(int(row["baseline_segments"]) for row in rows),
            "learned_segments": sum(int(row["learned_segments"]) for row in rows),
            "mean_learned_window_seconds": f"{sum(float(row['learned_window_seconds']) for row in rows)/len(rows):.3f}",
        })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "source_window_lomo_summary_v1.csv", summary_rows)
    write_csv(args.output / "source_window_lomo_by_match_v1.csv", match_rows)
    write_csv(args.output / "source_window_lomo_development_offsets_v1.csv", offset_rows)
    manifest = {
        "experiment_id": args.output.name,
        "scope": "TRAIN-only LOMO; 22 heldout match folds; 5,372 common groups; 130 goals; no VALIDATION/TEST or V4 promotion",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]],
        "sources": list(SOURCES), "k_per_match": K, "gap_seconds": GAP_SECONDS,
        "development_rule": "For each heldout match, fit P05/P95 of goal_timestamp minus the nearest candidate anchor among other 21 matches, per source; each source ranking uses its frozen LOMO scores.",
        "rescue_rule": "Temporal Top-35 core, then up to 10 nonduplicate fusion and 10 nonduplicate net_motion additions with 45s gap, matching A2; this policy may yield 55 candidates.",
        "interval_rule": "Uniform comparator is anchor-20s to anchor+40s; learned interval is anchor+P05(delta) to anchor+P95(delta), clipped at video start only.",
        "quantiles": [Q_LOW, Q_HIGH],
        "interpretation": "Exploratory TRAIN leave-one-match-out development; quantile and source choices require independent confirmation.",
    }
    (args.output / "manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(SOURCES), "heldout_rows": len(match_rows), "offset_rows": len(offset_rows), "summary_rows": len(summary_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
