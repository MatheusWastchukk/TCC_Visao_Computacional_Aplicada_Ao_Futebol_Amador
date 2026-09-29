"""Audit multiplicidade de propostas BALL nos frames candidato TRAIN já persistidos."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES = sorted((ROOT / "experiments/ball/GOAL-EVENT-CANDIDATE-MINER-V1").rglob("candidate_frames_v1.csv"))
DEFAULT_GOALS = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/train_goal_pool_and_shortlist_coverage_v1.csv"
DEFAULT_MANUAL = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/manual_review_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/GOAL-EVENT-BALL-DETECTION-MULTIPLICITY-AUDIT-TRAIN-V1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--goals", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--manual-audit", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--miss-window-before", type=float, default=25.0)
    parser.add_argument("--miss-window-after", type=float, default=35.0)
    parser.add_argument("--control-goal-exclusion", type=float, default=60.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output already contains files: {args.output}")
    per_frame: dict[tuple[str, int], dict[str, object]] = {}
    source_stats = []
    for path in args.frames:
        rows = read_csv(path)
        if not rows:
            continue
        match_set = {r["match_id"] for r in rows}
        if len(match_set) != 1:
            # Some sequential mining outputs contain multiple matches; partition below.
            pass
        for row in rows:
            key = (row["match_id"], int(row["frame_index"]))
            record = per_frame.setdefault(key, {"match_id": row["match_id"], "frame_index": int(row["frame_index"]),
                "timestamp_seconds": float(row["timestamp_seconds"]), "ball_boxes": set(), "confidences": [],
                "source_files": set()})
            box = tuple(round(float(x), 2) for x in json.loads(row["ball_box_json"]))
            record["ball_boxes"].add(box)
            record["confidences"].append(float(row["ball_confidence"]))
            record["source_files"].add(str(path))
        source_stats.append({"source_file": str(path), "rows": len(rows), "matches": sorted(match_set)})

    frame_rows: list[dict[str, object]] = []
    for record in per_frame.values():
        if len(record["source_files"]) != 1:
            raise ValueError(f"Frame duplicated across source CSVs: {record['match_id']} {record['frame_index']}")
        frame_rows.append({"match_id": record["match_id"], "frame_index": record["frame_index"],
            "timestamp_seconds": f"{record['timestamp_seconds']:.3f}",
            "ball_candidate_rows": len(record["confidences"]), "unique_ball_boxes": len(record["ball_boxes"]),
            "max_ball_confidence": f"{max(record['confidences']):.6f}"})
    frame_rows.sort(key=lambda r: (str(r["match_id"]), int(r["frame_index"])))

    goals = read_csv(args.goals)
    if any(g.get("split", "").lower() != "train" for g in goals):
        raise AssertionError("TRAIN-only catalog required; TEST must remain closed.")
    by_match_goals: dict[str, list[float]] = defaultdict(list)
    for goal in goals:
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds"):
            by_match_goals[goal["match_id"]].append(float(goal["event_timestamp_seconds"]))
    per_match_frames: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame_rows:
        per_match_frames[str(row["match_id"])].append(row)
    if len(per_match_frames) != 25 or len(frame_rows) < 1000 or sum(map(len, by_match_goals.values())) != 152:
        raise ValueError(f"Unexpected TRAIN scope: {len(per_match_frames)} matches, {len(frame_rows)} frames, {sum(map(len, by_match_goals.values()))} goals")

    summary_rows: list[dict[str, object]] = []
    for match, rows in sorted(per_match_frames.items()):
        for stratum in ("within_60s_of_catalog_goal", "outside_60s_of_catalog_goals"):
            selected = [r for r in rows if
                (min(abs(float(r["timestamp_seconds"]) - t) for t in by_match_goals[match]) <= args.control_goal_exclusion)
                == (stratum == "within_60s_of_catalog_goal")]
            counts = [int(r["unique_ball_boxes"]) for r in selected]
            if not counts:
                continue
            summary_rows.append({"match_id": match, "stratum": stratum, "candidate_positive_frames": len(counts),
                "mean_unique_ball_boxes": f"{sum(counts)/len(counts):.4f}", "median_unique_ball_boxes": median(counts),
                "share_frames_multi_box": f"{sum(n > 1 for n in counts)/len(counts):.6f}",
                "share_frames_5plus_boxes": f"{sum(n >= 5 for n in counts)/len(counts):.6f}",
                "max_unique_ball_boxes": max(counts)})
    audit = read_csv(args.manual_audit)
    miss_rows: list[dict[str, object]] = []
    for item in audit:
        match, center = item["match_id"], float(item["goal_timestamp_seconds"])
        within = [r for r in per_match_frames[match] if center - args.miss_window_before <= float(r["timestamp_seconds"]) <= center + args.miss_window_after]
        counts = [int(r["unique_ball_boxes"]) for r in within]
        miss_rows.append({"goal_catalog_id": item["goal_catalog_id"], "match_id": match,
            "goal_timestamp_seconds": f"{center:.3f}", "analysis_window_seconds": f"-{args.miss_window_before}/+{args.miss_window_after}",
            "candidate_positive_frames": len(counts), "median_unique_ball_boxes": median(counts) if counts else "",
            "share_frames_multi_box": f"{sum(n > 1 for n in counts)/len(counts):.6f}" if counts else "",
            "share_frames_5plus_boxes": f"{sum(n >= 5 for n in counts)/len(counts):.6f}" if counts else "",
            "max_unique_ball_boxes": max(counts) if counts else ""})
    all_counts = [int(r["unique_ball_boxes"]) for r in frame_rows]
    aggregate = {"candidate_positive_frames": len(frame_rows), "mean_unique_ball_boxes": sum(all_counts)/len(all_counts),
        "median_unique_ball_boxes": median(all_counts), "share_frames_multi_box": sum(n > 1 for n in all_counts)/len(all_counts),
        "share_frames_5plus_boxes": sum(n >= 5 for n in all_counts)/len(all_counts), "max_unique_ball_boxes": max(all_counts)}
    args.output.mkdir(parents=True)
    write_csv(args.output / "frame_ball_multiplicity_v1.csv", frame_rows)
    write_csv(args.output / "multiplicity_by_match_stratum_v1.csv", summary_rows)
    write_csv(args.output / "audited_miss_multiplicity_v1.csv", miss_rows)
    summary = {"experiment_id": args.output.name, "scope": "TRAIN only; 25 matches, 152 catalog goals; candidate-positive sampled frames from saved miner detections; TEST closed",
        "source_files": source_stats, "aggregate": aggregate,
        "filters_and_limits": ["Miner writes only frames where GOAL detections exist.",
            "Only BALL detections within the miner's expanded-goal / max-normalized-distance gate are stored.",
            "Counts therefore describe ball proposals near detected goal structure, not every BALL box in the full frame.",
            "No detection means no row and cannot be counted as zero from these files."],
        "miss_window": {"before_seconds": args.miss_window_before, "after_seconds": args.miss_window_after},
        "control_strata": {"goal_near_seconds": args.control_goal_exclusion, "note": "Control is candidate-positive frames outside any catalog-goal exclusion interval; not all sampled video frames."},
        "warning": "Descriptive detector-output audit only; manual miss annotations are joined only after aggregation for the case-level table. No model training or ranking selection. Existing detections were generated in TRAIN; no inference rerun."}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": summary["scope"], "aggregate": aggregate,
        "match06_goal004": next(r for r in miss_rows if r["goal_catalog_id"] == "match_06__catalog_goal_004")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
