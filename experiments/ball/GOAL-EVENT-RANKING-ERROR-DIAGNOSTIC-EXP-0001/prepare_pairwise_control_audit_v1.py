"""Prepara até um candidato comparador de maior rank por gol perdido, sem sobreposições conhecidas com o catálogo."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"


def read_csv(name: str) -> list[dict[str, str]]:
    with (EXP / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    pairs = read_csv("pairwise_rank_diagnostic.csv")
    trace = read_csv("selection_trace.csv")
    goals = read_csv("case_manifest.csv")
    misses = read_csv("timestamp_rank_miss_diagnostic.csv")
    trace_by_group = {row["group_uid"]: row for row in trace}
    pairs_by_goal: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pairs:
        pairs_by_goal[row["goal_catalog_id"]].append(row)
    for row in goals:
        goals_by_match[row["match_id"]].append(row)

    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()
    no_control: list[str] = []
    for miss in sorted(misses, key=lambda row: (row["match_id"], float(row["goal_timestamp_seconds"]))):
        catalog_id = miss["goal_catalog_id"]
        goal_rank = int(miss["best_temporal_rank_full_pool"])
        options = sorted(pairs_by_goal.get(catalog_id, []), key=lambda row: (int(row["candidate_temporal_rank_full_pool"]), row["candidate_group_uid"]))
        chosen = None
        for pair in options:
            uid = pair["candidate_group_uid"]
            candidate = trace_by_group.get(uid)
            if not candidate or uid in used_groups:
                continue
            candidate_rank = int(candidate["temporal_rank_full_pool"])
            if candidate_rank >= goal_rank:
                continue
            start, end = float(candidate["delivery_start_seconds"]), float(candidate["delivery_end_seconds"])
            catalog_hits = [g["goal_catalog_id"] for g in goals_by_match[miss["match_id"]]
                            if start <= float(g["goal_timestamp_seconds"]) <= end]
            if catalog_hits:
                continue
            chosen = (pair, candidate)
            break
        if not chosen:
            no_control.append(catalog_id)
            continue
        pair, candidate = chosen
        used_groups.add(candidate["group_uid"])
        selected.append({
            "goal_catalog_id": catalog_id,
            "match_id": miss["match_id"],
            "video_filename": candidate["video_filename"],
            "goal_timestamp_seconds": miss["goal_timestamp_seconds"],
            "missed_goal_group_uid": miss["best_group_uid"],
            "missed_goal_rank_full_pool": str(goal_rank),
            "control_group_uid": candidate["group_uid"],
            "control_candidate_timestamp_seconds": candidate["best_timestamp_seconds"],
            "control_rank_full_pool": str(candidate_rank),
            "control_temporal_score_lomo": candidate["temporal_score_lomo"],
            "delivery_start_seconds": candidate["delivery_start_seconds"],
            "delivery_end_seconds": candidate["delivery_end_seconds"],
            "candidate_frames_csv": candidate.get("candidate_frames_csv", ""),
            "weak_train_label_unconfirmed": candidate["weak_train_label_unconfirmed"],
            "manual_label": "",
            "manual_status": "PENDING",
            "reviewer_note": "",
        })

    destination = EXP / "pairwise_control_audit_v1.csv"
    fields = list(selected[0]) if selected else []
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(f"misses={len(misses)} unique_controls={len(selected)} no_eligible_control={len(no_control)} output={destination}")
    if no_control:
        print("sem comparador elegível: " + ", ".join(no_control))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
