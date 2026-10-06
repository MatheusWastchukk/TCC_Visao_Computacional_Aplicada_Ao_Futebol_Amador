"""One-shot confirmation of the frozen B4 tiered-window policy on VALIDATION."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(rows_by_match: dict[str, list[dict[str, object]]], goals: dict[str, list[dict[str, str]]], variant: str, windows: dict[str, int]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_match: list[dict[str, object]] = []
    selected_out: list[dict[str, object]] = []
    goal_out: list[dict[str, object]] = []
    total_covered = total_goals = total_segments = 0
    total_seconds = 0.0
    for match, pool in sorted(rows_by_match.items()):
        ranked: list[dict[str, object]] = []
        source_ranks: dict[str, dict[str, int]] = {}
        for source in ("rank_score", "visual_score_goal", "fusion_score_goal", "net_motion"):
            order = sorted(pool, key=lambda row: (-float(row[source]), str(row["group_uid"])))
            source_ranks[source] = {str(row["group_uid"]): i for i, row in enumerate(order, 1)}
        rrf: dict[str, float] = defaultdict(float)
        for ranks in source_ranks.values():
            for uid, rank in ranks.items():
                rrf[uid] += 1.0 / (60 + rank)
        ordered = sorted(pool, key=lambda row: (-rrf[str(row["group_uid"])], str(row["group_uid"])))
        for row in ordered:
            uid = str(row["group_uid"])
            support = sum(source_ranks[name][uid] <= 60 for name in source_ranks)
            temporal_rank = source_ranks["rank_score"][uid]
            row["tier"] = "A" if support >= 3 or temporal_rank <= 35 else ("B" if support == 2 else "C")
            row["support_count_top60"] = support
            row["temporal_rank"] = temporal_rank
        chosen: list[dict[str, object]] = []
        for row in ordered:
            center = float(row["delivery_start_seconds"]) + 20.0
            if all(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) >= 45.0 for other in chosen):
                chosen.append(row)
            if len(chosen) == 60:
                break
        delivered: list[dict[str, object]] = []
        intervals: list[tuple[float, float]] = []
        for rank, row in enumerate(chosen, 1):
            tier = str(row["tier"])
            width = windows[tier]
            anchor = float(row["delivery_start_seconds"]) + 20.0
            start = max(0.0, anchor - width / 3.0)
            end = anchor + (width - width / 3.0)
            item = {"variant": variant, "match_id": match, "group_uid": row["group_uid"], "rrf_rank": rank,
                    "tier": tier, "support_count_top60": row["support_count_top60"], "temporal_rank": row["temporal_rank"],
                    "window_seconds": width, "delivery_start_seconds": f"{start:.3f}", "delivery_end_seconds": f"{end:.3f}"}
            delivered.append(item)
            selected_out.append(item)
            intervals.append((start, end))
        covered = sum(any(start <= float(goal["event_timestamp_seconds"]) <= end for start, end in intervals) for goal in goals[match])
        for goal in goals[match]:
            goal_out.append({"variant": variant, "match_id": match, "goal_catalog_id": goal.get("goal_catalog_id", ""),
                             "goal_timestamp_seconds": goal["event_timestamp_seconds"],
                             "covered": any(start <= float(goal["event_timestamp_seconds"]) <= end for start, end in intervals)})
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        seconds = sum(end - start for start, end in merged)
        per_match.append({"variant": variant, "match_id": match, "catalog_goals": len(goals[match]), "covered_goals": covered,
                          "selected_candidates": len(delivered), "merged_segments": len(merged), "unique_review_minutes": f"{seconds/60:.3f}"})
        total_goals += len(goals[match]); total_covered += covered; total_segments += len(merged); total_seconds += seconds
    summary = {"variant": variant, "catalog_goals": total_goals, "covered_goals": total_covered,
               "recall": f"{total_covered/total_goals:.6f}", "selected_candidates": sum(int(r["selected_candidates"]) for r in per_match),
               "merged_segments": total_segments, "unique_review_minutes_per_match": f"{total_seconds/60/len(per_match):.3f}"}
    return summary, per_match, selected_out, goal_out


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, default=root / "results/net_motion_dataset_train_v1/temporal_group_dataset_net_motion_v1.csv")
    parser.add_argument("--validation-shot-dataset", type=Path, default=root / "results/shot_motion_dataset_validation_v1/temporal_group_dataset_shot_motion_v1.csv")
    parser.add_argument("--validation-net-features", type=Path, default=root / "results/net_motion_features_validation_v1/net_motion_groups_v1.csv")
    parser.add_argument("--candidate-scores", type=Path, default=root / "experiments/ball/GOAL-EVENT-FULL-POOL-VALIDATION-V1/fusion_scored_v1/candidate_fusion_scores_v1.csv")
    parser.add_argument("--goal-timestamps", type=Path, default=root / "experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-VALIDATION-V1/goal_catalog_timestamp_manifest_v1_annotated.csv")
    parser.add_argument("--output", type=Path, default=root / "experiments/ball/TIERED-WINDOW-VALIDATION-CONFIRMATION-V1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train, train_fields = read_csv(args.train_dataset)
    validation, validation_fields = read_csv(args.validation_shot_dataset)
    net, net_fields = read_csv(args.validation_net_features)
    candidates, candidate_fields = read_csv(args.candidate_scores)
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    for rows, label in ((candidates, "candidate scores"), (catalog, "catalog")):
        if not rows or any(row.get("split", "").lower() != "validation" for row in rows):
            raise AssertionError(f"{label} must contain VALIDATION only")
    if any(row.get("split", "").lower() != "validation" for row in catalog):
        raise AssertionError("VALIDATION goal catalog required")
    excluded = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    features = [field for field in train_fields if field not in excluded]
    if len(train) != 5372 or len(features) != 52:
        raise ValueError(f"Unexpected frozen TRAIN contract: {len(train)} rows / {len(features)} features")
    net_by_uid = {row["group_uid"]: row for row in net}
    if len(net_by_uid) != len(net) or set(net_fields) < {"group_uid", "net_motion_inside_mean", "net_motion_ring_mean", "net_motion_ratio"}:
        raise ValueError("Invalid VALIDATION net motion features")
    validation_by_uid = {row["group_uid"]: row for row in validation}
    if len(validation_by_uid) != 1944 or set(validation_by_uid) != set(net_by_uid):
        raise ValueError("VALIDATION net motion and shot-motion groups do not align one-to-one")
    net_rename = {"net_net_motion_inside_mean": "net_motion_inside_mean", "net_net_motion_ring_mean": "net_motion_ring_mean", "net_net_motion_ratio": "net_motion_ratio"}
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    x_train = np.asarray([[float(row[f]) for f in features] for row in train], dtype=np.float32)
    y_train = np.asarray([int(row["weak_label"]) for row in train], dtype=np.int32)
    model.fit(x_train, y_train)
    rows_scored: list[dict[str, object]] = []
    required_scores = {"group_uid", "match_id", "delivery_start_seconds", "delivery_end_seconds", "rank_score", "visual_score_goal", "fusion_score_goal", "split"}
    if required_scores - set(candidate_fields):
        raise ValueError(f"Missing candidate score fields: {sorted(required_scores-set(candidate_fields))}")
    candidate_by_uid = {row["group_uid"]: row for row in candidates}
    if len(candidate_by_uid) != 1944 or len(candidate_by_uid) != len(candidates):
        raise ValueError("Expected exactly 1,944 unique full-pool VALIDATION candidates")
    assembled_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    for uid, row in validation_by_uid.items():
        assembled = {**row, **{name: net_by_uid[uid][source] for name, source in net_rename.items()}}
        if set(features) - set(assembled):
            raise ValueError(f"VALIDATION feature contract missing fields for {uid}")
        source_candidate = candidate_by_uid[uid]
        if source_candidate["match_id"] != row["match_id"]:
            raise ValueError(f"Match mismatch for {uid}")
        assembled_rows.append(assembled)
        candidate_rows.append(source_candidate)
    x_validation = np.asarray([[float(row[f]) for f in features] for row in assembled_rows], dtype=np.float32)
    scores = model.predict_proba(x_validation)[:, list(model.classes_).index(1)]
    rows_scored = [{**row, "net_motion": float(score)} for row, score in zip(candidate_rows, scores)]
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows_scored:
        by_match[str(row["match_id"])].append(row)
    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(row)
    if set(by_match) != set(goals) or len(by_match) != 8 or not sum(map(len, goals.values())):
        raise ValueError(f"Expected the same 8 annotated VALIDATION matches; got {len(by_match)} / {sum(map(len,goals.values()))}")
    summaries: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    goal_rows: list[dict[str, object]] = []
    for variant, windows in (("uniform_60_60_60", {"A": 60, "B": 60, "C": 60}), ("tier_40_45_60", {"A": 40, "B": 45, "C": 60})):
        summary, match_records, chosen_records, goal_records = measure(by_match, goals, variant, windows)
        summaries.append(summary); per_match_rows.extend(match_records); selected_rows.extend(chosen_records); goal_rows.extend(goal_records)
    args.output.mkdir(parents=True)
    write_csv(args.output / "tiered_window_validation_summary_v1.csv", summaries)
    write_csv(args.output / "tiered_window_validation_per_match_v1.csv", per_match_rows)
    write_csv(args.output / "tiered_window_validation_selected_v1.csv", selected_rows)
    write_csv(args.output / "tiered_window_validation_goal_coverage_v1.csv", goal_rows)
    write_csv(args.output / "validation_candidate_scores_net_motion_v1.csv", rows_scored)
    inputs = [args.train_dataset, args.validation_shot_dataset, args.validation_net_features, args.candidate_scores, args.goal_timestamps]
    metadata = {"experiment_id": args.output.name, "scope": "one-shot VALIDATION confirmation; TRAIN-only RandomForest net_motion; no TEST access",
                "selection_frozen_before_validation": {"k": 60, "gap_seconds": 45, "tier_rule": "A=support>=3/4 source Top-60 OR temporal rank<=35; B=exactly 2 source Top-60; C=otherwise", "window_candidate": "A/B/C=40/45/60 seconds; centered using pre=1/3 and post=2/3", "reference": "uniform 60/60/60"},
                "net_motion_model": "RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight=balanced,random_state=42) trained on TRAIN net-motion features only",
                "validation_labels_used_only_for_final_scoring": True, "inputs_sha256": {str(p.relative_to(root)): sha256(p) for p in inputs}}
    (args.output / "summary_v1.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
