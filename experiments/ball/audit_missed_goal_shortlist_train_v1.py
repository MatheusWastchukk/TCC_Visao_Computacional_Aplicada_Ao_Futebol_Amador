"""Auditable per-goal decomposition for the TRAIN RRF+NMS IoU .10 Top-60 result."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SCORES = ("rank_score", "visual_score_goal", "fusion_score_goal")
RRF_CONSTANT = 60
NMS_IOU = 0.10
SHORTLIST_K = 60


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval_iou(a: dict[str, str], b: dict[str, str]) -> float:
    a0, a1 = float(a["delivery_start_seconds"]), float(a["delivery_end_seconds"])
    b0, b1 = float(b["delivery_start_seconds"]), float(b["delivery_end_seconds"])
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - intersection
    return intersection / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    candidates, fields = read_csv(args.candidate_scores)
    required = {"candidate_id", "match_id", "video_filename", "group_uid", "delivery_start_seconds", "delivery_end_seconds", *SCORES}
    if missing := required - set(fields):
        raise ValueError(f"Candidate scores missing columns: {sorted(missing)}")
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"match_id", "video_filename", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Goal catalog missing columns: {sorted(missing)}")
    if not catalog or any(r["split"].lower() != "train" for r in catalog):
        raise AssertionError("Goal catalog must contain TRAIN only.")

    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(row)
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_match[row["match_id"]].append(row)
    if set(by_match) != set(goals_by_match) or len(candidates) != 5742:
        raise ValueError("Expected the complete 5,742-group TRAIN pool and matching annotated TRAIN matches.")

    all_goals: list[dict[str, str]] = []
    shortlist_by_match: dict[str, list[dict[str, str]]] = {}
    nms_ranks: dict[tuple[str, str], int] = {}
    suppressed_by: dict[tuple[str, str], dict[str, str]] = {}
    ranking_maps: dict[str, dict[tuple[str, str], int]] = {field: {} for field in SCORES}
    rrf_rank_map: dict[tuple[str, str], int] = {}
    coverage_by_goal: dict[str, list[dict[str, str]]] = {}

    for match, rows in by_match.items():
        for field in SCORES:
            ordered = sorted(rows, key=lambda r: (-float(r[field]), r["group_uid"]))
            for rank, row in enumerate(ordered, 1):
                ranking_maps[field][(match, row["group_uid"])] = rank
        rrf: dict[str, float] = defaultdict(float)
        for field in SCORES:
            for rank, row in enumerate(sorted(rows, key=lambda r: (-float(r[field]), r["group_uid"])), 1):
                rrf[row["group_uid"]] += 1.0 / (RRF_CONSTANT + rank)
        rrf_order = sorted(rows, key=lambda r: (-rrf[r["group_uid"]], r["group_uid"]))
        for rank, row in enumerate(rrf_order, 1):
            rrf_rank_map[(match, row["group_uid"])] = rank

        kept: list[dict[str, str]] = []
        for row in rrf_order:
            suppressor = next((other for other in kept if interval_iou(row, other) > NMS_IOU), None)
            if suppressor is None:
                kept.append(row)
                nms_ranks[(match, row["group_uid"])] = len(kept)
            else:
                suppressed_by[(match, row["group_uid"])] = suppressor
        shortlist = kept[:SHORTLIST_K]
        shortlist_by_match[match] = shortlist

        for goal in goals_by_match[match]:
            timestamp = float(goal["event_timestamp_seconds"])
            covering = [r for r in rows if float(r["delivery_start_seconds"]) <= timestamp <= float(r["delivery_end_seconds"])]
            coverage_by_goal[goal["goal_catalog_id"]] = covering
            is_shortlisted = any(r in shortlist and float(r["delivery_start_seconds"]) <= timestamp <= float(r["delivery_end_seconds"]) for r in rows)
            item = {**goal, "is_pool_recoverable": str(bool(covering)).lower(), "is_covered_by_rrf_nms_top60": str(bool(is_shortlisted)).lower()}
            all_goals.append(item)

    audit_rows: list[dict[str, object]] = []
    for goal in all_goals:
        match, goal_id = goal["match_id"], goal["goal_catalog_id"]
        if goal["is_covered_by_rrf_nms_top60"] == "true":
            continue
        covering = coverage_by_goal[goal_id]
        representative = min(covering, key=lambda r: (rrf_rank_map[(match, r["group_uid"])], r["group_uid"])) if covering else None
        key = (match, representative["group_uid"]) if representative else None
        suppressor = suppressed_by.get(key) if key else None
        nms_rank = nms_ranks.get(key) if key else None
        if not covering:
            disposition = "unrecoverable_pool_miss"
        elif suppressor:
            disposition = "suppressed_by_nms"
        elif nms_rank and nms_rank > SHORTLIST_K:
            disposition = "outside_top60_after_nms"
        elif nms_rank:
            disposition = "alternate_cover_exists_but_not_selected"
        else:
            disposition = "requires_review"

        row: dict[str, object] = {
            "goal_catalog_id": goal_id,
            "match_id": match,
            "video_filename": goal["video_filename"],
            "goal_timestamp_seconds": goal["event_timestamp_seconds"],
            "audit_status": disposition,
            "pool_recoverable": bool(covering),
            "candidate_groups_covering_goal": len(covering),
            "representative_candidate_id": representative["candidate_id"] if representative else "",
            "representative_group_uid": representative["group_uid"] if representative else "",
            "representative_best_timestamp_seconds": (
                f"{float(representative['candidate_start_seconds']) + 10.0:.3f}"
                if representative and "candidate_start_seconds" in fields else ""
            ),
            "rank_temporal": ranking_maps["rank_score"].get(key, "") if key else "",
            "score_temporal": representative["rank_score"] if representative else "",
            "rank_visual": ranking_maps["visual_score_goal"].get(key, "") if key else "",
            "score_visual": representative["visual_score_goal"] if representative else "",
            "rank_fusion": ranking_maps["fusion_score_goal"].get(key, "") if key else "",
            "score_fusion": representative["fusion_score_goal"] if representative else "",
            "rank_rrf": rrf_rank_map.get(key, "") if key else "",
            "nms_rank_if_kept": nms_rank or "",
            "nms_suppressor_candidate_id": suppressor["candidate_id"] if suppressor else "",
            "nms_suppressor_group_uid": suppressor["group_uid"] if suppressor else "",
            "nms_suppressor_rrf_rank": rrf_rank_map.get((match, suppressor["group_uid"]), "") if suppressor else "",
            "nms_suppressor_iou": f"{interval_iou(representative, suppressor):.4f}" if representative and suppressor else "",
            "covering_candidate_ids": ";".join(r["candidate_id"] for r in sorted(covering, key=lambda x: rrf_rank_map[(match, x["group_uid"])])),
            "manual_ball_visible": "",
            "manual_clear_offensive_action": "",
            "manual_goal_visible": "",
            "manual_tracking_problem": "",
            "manual_notes": "Pending human video review",
        }
        audit_rows.append(row)

    pool_recoverable = sum(row["is_pool_recoverable"] == "true" for row in all_goals)
    covered = sum(row["is_covered_by_rrf_nms_top60"] == "true" for row in all_goals)
    args.output.mkdir(parents=True)
    write_csv(args.output / "train_missed_goal_audit_v1.csv", audit_rows)
    write_csv(args.output / "train_goal_pool_and_shortlist_coverage_v1.csv", all_goals)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 25-fold LOMO scores; TEST not accessed",
        "candidate_rows": len(candidates),
        "catalog_goals": len(all_goals),
        "pool_recoverable_goals": pool_recoverable,
        "unrecoverable_pool_misses": len(all_goals) - pool_recoverable,
        "rrf_nms_iou": NMS_IOU,
        "shortlist_k": SHORTLIST_K,
        "covered_by_shortlist": covered,
        "recoverable_but_missed": pool_recoverable - covered,
        "audit_rows": len(audit_rows),
        "audit_dispositions": dict(sorted({status: sum(r["audit_status"] == status for r in audit_rows) for status in {r["audit_status"] for r in audit_rows}}.items())),
        "warning": "Ranks and NMS causes are automatic. Visual/action/tracking fields are intentionally blank pending human video review.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
