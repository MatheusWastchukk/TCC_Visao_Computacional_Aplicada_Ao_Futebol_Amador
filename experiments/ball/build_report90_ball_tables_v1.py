"""Build separate proposal-availability and tracking tables for Report 90."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments/ball/REPORT90-BALL-TABLES-EXP-0001"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    ceiling_path = ROOT / "experiments/ball/BALL-PROPOSAL-CEILING-EXP-0001/proposal_ceiling_metrics_v1.csv"
    tiled_detection_path = ROOT / "experiments/ball/BALL-TILED-INFERENCE-EXP-0001/tiled_inference_metrics_v1.csv"
    tiled_track_path = ROOT / "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001/tiled_tracker_scored_summary_v1.csv"
    top_grid_path = ROOT / "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001/budget_grid_v1/tiled_tracker_budget_grid_summary_v1.csv"
    inputs = [ceiling_path, tiled_detection_path, tiled_track_path, top_grid_path]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    ceiling = read_csv(ceiling_path)
    tiled_detection = read_csv(tiled_detection_path)
    proposal_rows: list[dict[str, object]] = []
    for checkpoint in ("baseline", "oversampled"):
        for configuration, source, confidence, top_n in (
            ("global_current", ceiling, "0.01", "10"),
            ("global_permissive", ceiling, "0.001", "all"),
            ("global_plus_tiles_permissive", tiled_detection, "0.001", "all_views_post_nms"),
        ):
            if configuration == "global_plus_tiles_permissive":
                rows = [row for row in source if row["checkpoint"] == checkpoint and row["view"] == "global_plus_tiles" and row["confidence_threshold"] == confidence]
            else:
                rows = [row for row in source if row["checkpoint"] == checkpoint and row["confidence_threshold"] == confidence and row["top_n"] == top_n]
            if len(rows) != 1:
                raise ValueError(f"Expected one proposal row for {checkpoint}/{configuration}; got {len(rows)}")
            row = rows[0]
            proposal_rows.append({
                "configuration": configuration,
                "checkpoint": checkpoint,
                "split": "TRAIN frozen five-match holdout",
                "positive_frames": int(row["positive_frames"]),
                "confidence_threshold": confidence,
                "top_n": top_n,
                "candidate_hits_iou_050": int(row["hits_iou_050"]),
                "candidate_recall_iou_050": float(row["candidate_recall_iou_050"]),
                "candidate_hits_iou_030": int(row["hits_iou_030"]),
                "candidate_recall_iou_030": float(row["candidate_recall_iou_030"]),
                "mean_proposals_per_positive_frame": float(row.get("mean_proposals_per_positive_frame", row.get("mean_proposals_per_frame", ""))),
                "negative_frames_with_any_proposal": int(row.get("not_visible_negative_frames_with_proposal", "")) if row.get("not_visible_negative_frames_with_proposal") else "",
                "mean_inference_seconds_per_frame": float(row.get("mean_inference_seconds_per_frame", "")) if row.get("mean_inference_seconds_per_frame") else "",
                "source": str((tiled_detection_path if configuration == "global_plus_tiles_permissive" else ceiling_path).relative_to(ROOT)).replace("\\", "/"),
                "interpretation": "proposal availability only; not tracker identity or goal detection",
            })

    tracker_original = read_csv(tiled_track_path)
    tracker_grid = read_csv(top_grid_path)
    tracker_rows: list[dict[str, object]] = []
    original_map = {
        "global_baseline": ("baseline", "global_current"),
        "global_oversampled": ("oversampled", "global_current"),
    }
    for source_row in tracker_original:
        if source_row["run"] not in original_map or source_row["event_role"] not in {"", "CATALOGED_GOAL"}:
            continue
        checkpoint, configuration = original_map[source_row["run"]]
        target_frames = int(source_row["target_frames"])
        scope = source_row["scope"]
        role = source_row["event_role"]
        sequences = 35 if role == "" else 29
        frames_per_sequence = 11 if scope == "center_16_26" else 41
        tracker_rows.append({
            "configuration": configuration,
            "checkpoint": checkpoint,
            "scope": scope,
            "event_role": role or "ALL",
            "sequence_count": sequences,
            "frames_per_sequence": frames_per_sequence,
            "target_frames": target_frames,
            "candidate_hits": int(source_row["candidate_hits"]),
            "candidate_recall": float(source_row["candidate_recall"]),
            "identity_correct_frames": int(source_row["identity_correct_frames"]),
            "identity_accuracy": float(source_row["identity_accuracy"]),
            "wrong_ball_switch_frames": int(source_row["wrong_ball_switch_frames"]),
            "wrong_ball_switch_rate": float(source_row["wrong_ball_switch_rate"]),
            "mean_proposals_per_target_frame": float(source_row["mean_proposals_per_target_frame"]),
            "confidence_threshold": "0.01",
            "top_n": 10,
            "source": str(tiled_track_path.relative_to(ROOT)).replace("\\", "/"),
        })

    # Add the limited Top-3 vs Top-10 Viterbi comparison from saved tiled proposals.
    for row in tracker_grid:
        if row["confidence_floor"] != "0.010" or row["top_n"] not in {"3", "10"}:
            continue
        if row["event_role"] not in {"ALL", "CATALOGED_GOAL"}:
            continue
        scope = row["scope"]
        role = row["event_role"]
        if scope not in {"full", "center_16_26"}:
            continue
        if scope == "center_16_26" and role != "CATALOGED_GOAL":
            continue
        target_frames = int(row["target_frames"])
        tracker_rows.append({
            "configuration": f"global_plus_tiles_viterbi_top{row['top_n']}",
            "checkpoint": row["checkpoint"],
            "scope": scope,
            "event_role": role,
            "sequence_count": 29 if role == "CATALOGED_GOAL" else 35,
            "frames_per_sequence": 11 if scope == "center_16_26" else 41,
            "target_frames": target_frames,
            "candidate_hits": int(row["candidate_hits"]),
            "candidate_recall": float(row["candidate_recall"]),
            "identity_correct_frames": int(row["identity_correct_frames"]),
            "identity_accuracy": float(row["identity_accuracy"]),
            "wrong_ball_switch_frames": int(row["wrong_ball_switch_frames"]),
            "wrong_ball_switch_rate": float(row["wrong_ball_switch_rate"]),
            "mean_proposals_per_target_frame": float(row["mean_proposals_per_target_frame"]),
            "confidence_threshold": row["confidence_floor"],
            "top_n": int(row["top_n"]),
            "source": str(top_grid_path.relative_to(ROOT)).replace("\\", "/"),
        })
    tracker_rows.sort(key=lambda row: (str(row["scope"]), str(row["event_role"]), str(row["checkpoint"]), str(row["configuration"])))

    write_csv(output / "ball_proposal_availability_v1.csv", proposal_rows)
    write_csv(output / "ball_tracking_identity_v1.csv", tracker_rows)
    summary = {
        "experiment_id": "REPORT90-BALL-TABLES-EXP-0001",
        "scope": "separate summaries of frozen TRAIN proposal and sequence-tracking evaluations; no VALIDATION/TEST",
        "proposal_population": "104 VISIBLE/PARTIAL frames from five-match TRAIN holdout; IoU metrics apply to positive frames",
        "tracking_population": "35 sequences x 41 frames; 29 cataloged-goal sequences and 6 confirmed no-goal controls; center scope frames 16-26",
        "metric_population_warning": "proposal recall and sequence identity use different sample populations and must be reported in separate tables",
        "proposal_inputs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in (ceiling_path, tiled_detection_path)],
        "tracking_inputs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in (tiled_track_path, top_grid_path)],
        "interpretation": "tiled increases proposal availability; Top-3/Top-10 trading candidate recall against identity and switches remains exploratory and does not promote tracking to V4",
    }
    (output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"proposal_rows={len(proposal_rows)} tracking_rows={len(tracker_rows)}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
