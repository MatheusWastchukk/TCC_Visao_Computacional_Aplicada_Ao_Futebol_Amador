"""Prepare the next 30 human-review controls without assigning negative labels."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
TARGET = 30
GAP = 45.0


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--output", type=Path, default=EXP / "pairwise_control_audit_v2_batch1.csv")
    parser.add_argument("--exclude-controls", type=Path, action="append", default=[],
                        help="Previously reviewed control CSV; can be repeated.")
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target must be positive")

    pairs = read(EXP / "pairwise_rank_diagnostic.csv")
    trace = {row["group_uid"]: row for row in read(EXP / "selection_trace.csv")}
    goals = read(EXP / "case_manifest.csv")
    misses = read(EXP / "timestamp_rank_miss_diagnostic.csv")
    reviewed = read(EXP / "pairwise_control_audit_v1_annotated.csv")
    previous_rows = [row for row in reviewed if row.get("manual_status") == "COMPLETED"]
    for path in args.exclude_controls:
        if not path.is_absolute():
            path = EXP / path
        previous_rows.extend(read(path))
    previous = {row["control_group_uid"] for row in previous_rows}

    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in goals:
        goals_by_match[row["match_id"]].append(row)
    pairs_by_goal: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pairs:
        pairs_by_goal[row["goal_catalog_id"]].append(row)

    selected: list[dict[str, str]] = []
    used = set(previous)
    selected_by_match: dict[str, list[float]] = defaultdict(list)
    previous_intervals_by_match: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in previous_rows:
        match = row.get("match_id", "")
        if row.get("control_candidate_timestamp_seconds"):
            selected_by_match[match].append(float(row["control_candidate_timestamp_seconds"]))
        if row.get("delivery_start_seconds") and row.get("delivery_end_seconds"):
            previous_intervals_by_match[match].append((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])))
    no_candidates = 0
    for miss in sorted(misses, key=lambda r: (r["match_id"], float(r["goal_timestamp_seconds"]))):
        options = sorted(
            pairs_by_goal.get(miss["goal_catalog_id"], []),
            key=lambda r: (int(r["candidate_temporal_rank_full_pool"]), r["candidate_group_uid"]),
        )
        added_for_goal = 0
        for pair in options:
            uid = pair["candidate_group_uid"]
            candidate = trace.get(uid)
            if not candidate or uid in used:
                continue
            rank = int(candidate["temporal_rank_full_pool"])
            if rank >= int(miss["best_temporal_rank_full_pool"]):
                continue
            start, end = float(candidate["delivery_start_seconds"]), float(candidate["delivery_end_seconds"])
            if any(start <= float(g["goal_timestamp_seconds"]) <= end for g in goals_by_match[miss["match_id"]]):
                continue
            center = float(candidate["best_timestamp_seconds"])
            if any(abs(center - old) < GAP for old in selected_by_match[miss["match_id"]]):
                continue
            selected.append({
                "goal_catalog_id": miss["goal_catalog_id"],
                "match_id": miss["match_id"],
                "video_filename": candidate["video_filename"],
                "goal_timestamp_seconds": miss["goal_timestamp_seconds"],
                "missed_goal_group_uid": miss["best_group_uid"],
                "missed_goal_rank_full_pool": miss["best_temporal_rank_full_pool"],
                "control_group_uid": uid,
                "control_candidate_timestamp_seconds": candidate["best_timestamp_seconds"],
                "control_rank_full_pool": str(rank),
                "control_temporal_score_lomo": candidate["temporal_score_lomo"],
                "delivery_start_seconds": candidate["delivery_start_seconds"],
                "delivery_end_seconds": candidate["delivery_end_seconds"],
                "candidate_frames_csv": candidate.get("candidate_frames_csv", ""),
                "weak_train_label_unconfirmed": candidate["weak_train_label_unconfirmed"],
                "manual_label": "",
                "manual_status": "PENDING",
                "reviewer_note": "",
                "review_context": "ALTERNATE_CONTROL_FOR_MISSED_CATALOG_GOAL",
            })
            used.add(uid)
            selected_by_match[miss["match_id"]].append(center)
            added_for_goal += 1
            if len(selected) >= args.target:
                break
        if added_for_goal == 0:
            no_candidates += 1
        if len(selected) >= args.target:
            break

    # The pairwise miss diagnostic has a small supply of unused alternatives.
    # Fill the remainder with top-ranked candidate windows absent from the
    # catalog, paired to the nearest known goal in the same match. No label is
    # inferred from this absence; the user must review each original video.
    if len(selected) < args.target:
        selected_centers = {r["control_group_uid"] for r in selected}
        for candidate in sorted(trace.values(), key=lambda r: (int(r["temporal_rank_full_pool"]), r["group_uid"])):
            uid, match = candidate["group_uid"], candidate["match_id"]
            if uid in used or uid in selected_centers:
                continue
            rank = int(candidate["temporal_rank_full_pool"])
            if rank > 60:
                continue
            start, end = float(candidate["delivery_start_seconds"]), float(candidate["delivery_end_seconds"])
            if any(start <= float(g["goal_timestamp_seconds"]) <= end for g in goals_by_match[match]):
                continue
            center = float(candidate["best_timestamp_seconds"])
            if any(start < old_end and end > old_start for old_start, old_end in previous_intervals_by_match[match]):
                continue
            if any(abs(center - old_center) < GAP for old_center in selected_by_match[match]):
                continue
            if any(start < float(old["delivery_end_seconds"]) and end > float(old["delivery_start_seconds"])
                   for old in selected if old["match_id"] == match):
                continue
            positive_options = [g for g in goals_by_match[match]
                                if g.get("best_temporal_rank_full_pool")
                                and int(g["best_temporal_rank_full_pool"]) > rank]
            if not positive_options:
                continue
            positive = min(positive_options, key=lambda g: abs(float(g["goal_timestamp_seconds"]) - center))
            selected.append({
                "goal_catalog_id": positive["goal_catalog_id"],
                "match_id": match,
                "video_filename": candidate["video_filename"],
                "goal_timestamp_seconds": positive["goal_timestamp_seconds"],
                "missed_goal_group_uid": positive["best_temporal_group_uid"],
                "missed_goal_rank_full_pool": positive["best_temporal_rank_full_pool"],
                "control_group_uid": uid,
                "control_candidate_timestamp_seconds": candidate["best_timestamp_seconds"],
                "control_rank_full_pool": str(rank),
                "control_temporal_score_lomo": candidate["temporal_score_lomo"],
                "delivery_start_seconds": candidate["delivery_start_seconds"],
                "delivery_end_seconds": candidate["delivery_end_seconds"],
                "candidate_frames_csv": candidate.get("candidate_frames_csv", ""),
                "weak_train_label_unconfirmed": candidate["weak_train_label_unconfirmed"],
                "manual_label": "",
                "manual_status": "PENDING",
                "reviewer_note": "",
                "review_context": "TOP60_UNCOVERED_WINDOW_NEAREST_CATALOG_GOAL",
            })
            used.add(uid)
            selected_centers.add(uid)
            selected_by_match[match].append(center)
            if len(selected) >= args.target:
                break

    if not selected:
        raise RuntimeError("No eligible new controls found.")
    output = args.output if args.output.is_absolute() else EXP / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, selected)
    print(f"new_review_items={len(selected)}; prior_reviewed_excluded={len(previous)}; no_miss_match={no_candidates}; output={output}")
    print("All labels are PENDING. A missing catalog timestamp is not treated as NO_GOAL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
