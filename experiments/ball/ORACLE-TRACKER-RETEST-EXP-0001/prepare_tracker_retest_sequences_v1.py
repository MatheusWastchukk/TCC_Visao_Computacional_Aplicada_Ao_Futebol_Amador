"""Freeze and extract matched TRAIN goal/control clips for the conditional Viterbi retest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
DETECTOR_EXP = ROOT / "experiments/ball/BALL-HARD-POSITIVE-EXP-0001"
GOALS = ROOT / "experiments/ball/GOAL-EVENT-RETRIEVAL-CHARACTERIZATION-TRAIN-V1/goal_rank_and_clip_position_v1.csv"
CONTROLS = DETECTOR_EXP / "tracking_candidate_controls_v1.csv"
VIDEOS = ROOT / "data/Videos_Edited"
FRAMES_DIR = EXP / "frames_v1"
EVENTS_OUT = EXP / "event_manifest_v1.csv"
ANNOTATIONS_OUT = EXP / "frame_annotations_v1.csv"
PROTOCOL_OUT = EXP / "protocol_v1.json"
SECONDS_BEFORE = 4.0
SECONDS_AFTER = 4.0
SAMPLE_FPS = 5.0
REUSE_TOLERANCE_SECONDS = 0.05

FIELDS = [
    "sequence_id", "sequence_frame", "sample_id", "event_id", "event_role", "pair_id",
    "matched_positive_id", "is_event_frame", "match_id", "video_filename", "capture_domain",
    "split", "timestamp_seconds", "timestamp_hhmmss", "image_path", "ball_visibility",
    "ball_of_play_box_json", "distractor_boxes_json", "goal_region_box_json",
    "goal_region_points_json", "goal_region_status", "goal_volume_points_json",
    "goal_volume_status", "annotation_status", "annotator", "annotation_source", "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    # The existing OpenCV annotation tools expect UTF-8 without a BOM.
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def format_time(seconds: float) -> str:
    value = max(0, round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def video_name(match_id: str) -> str:
    return f"Match_{int(match_id.split('_')[1]):02d}.mp4"


def build_events() -> tuple[list[dict[str, str]], dict[str, str]]:
    goal_rows = read_csv(GOALS)
    unique_goals: dict[str, dict[str, str]] = {}
    for row in goal_rows:
        if row["match_id"] not in {"match_10", "match_16", "match_23", "match_27", "match_36"}:
            continue
        unique_goals[row["goal_catalog_id"]] = row
    goals = [
        {
            "event_id": goal_id,
            "event_role": "CATALOGED_GOAL",
            "match_id": row["match_id"],
            "timestamp_seconds": f"{float(row['goal_timestamp_seconds']):.6f}",
            "source": str(GOALS.relative_to(ROOT)),
        }
        for goal_id, row in unique_goals.items()
    ]
    controls = [
        row for row in read_csv(CONTROLS)
        if row["match_id"] in {"match_10", "match_16", "match_23", "match_27", "match_36"}
        and row["manual_status"] == "COMPLETED"
        and row["manual_label"] == "NO_GOAL"
        and row["tracking_candidate_status"] == "CONFIRMED_NO_GOAL_CANDIDATE"
    ]
    if len(goals) != 29 or len(controls) != 6:
        raise ValueError(f"Expected 29 distinct catalog goals and 6 human-confirmed controls; got {len(goals)} and {len(controls)}")

    # Pair each confirmed control with one unique nearest goal from the same match.
    unused_goals = {match: [goal for goal in goals if goal["match_id"] == match] for match in {g["match_id"] for g in goals}}
    pair_by_goal: dict[str, str] = {}
    pair_by_control: dict[str, tuple[str, str]] = {}
    for control in sorted(controls, key=lambda row: (row["match_id"], float(row["control_candidate_timestamp_seconds"]))):
        match = control["match_id"]
        timestamp = float(control["control_candidate_timestamp_seconds"])
        candidates = unused_goals.get(match, [])
        if not candidates:
            raise ValueError(f"No unique goal remains to match confirmed control {control['control_group_uid']}")
        goal = min(candidates, key=lambda item: abs(float(item["timestamp_seconds"]) - timestamp))
        candidates.remove(goal)
        pair_id = re.sub(r"[^A-Za-z0-9_-]+", "_", control["control_group_uid"])
        pair_by_goal[goal["event_id"]] = pair_id
        pair_by_control[control["control_group_uid"]] = (pair_id, goal["event_id"])

    events: list[dict[str, str]] = []
    for goal in goals:
        pair_id = pair_by_goal.get(goal["event_id"], "")
        events.append({**goal, "pair_id": pair_id, "matched_positive_id": goal["event_id"] if pair_id else "", "source_batch": "goal_catalog"})
    for control in controls:
        pair_id, matched_positive_id = pair_by_control[control["control_group_uid"]]
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", control["control_group_uid"])
        events.append({
            "event_id": f"{safe_id}_NO_GOAL",
            "event_role": "CONFIRMED_NO_GOAL",
            "match_id": control["match_id"],
            "timestamp_seconds": f"{float(control['control_candidate_timestamp_seconds']):.6f}",
            "source": str(CONTROLS.relative_to(ROOT)),
            "pair_id": pair_id,
            "matched_positive_id": matched_positive_id,
            "source_batch": "prior50_confirmed",
        })
    events.sort(key=lambda item: (item["match_id"], float(item["timestamp_seconds"]), item["event_role"], item["event_id"]))
    return events, {"goal_catalog": sha256(GOALS), "confirmed_controls": sha256(CONTROLS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--manifest", type=Path, default=EVENTS_OUT)
    parser.add_argument("--output", type=Path, default=ANNOTATIONS_OUT)
    args = parser.parse_args()

    gate_path = DETECTOR_EXP / "holdout_evaluation_v1/tracker_gate_decision_v1.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("gate", {}).get("passed"):
        raise RuntimeError("The preregistered detector gate did not pass; tracker retest is not authorized.")

    events, source_hashes = build_events()
    previous = {row["sample_id"]: row for row in read_csv(args.output)} if args.output.is_file() else {}
    prior_path = ROOT / "experiments/ball/BALL-EVENT-TRAJECTORY-DATASET-V1/event_trajectory_sequences_v1.csv"
    prior_rows = read_csv(prior_path) if prior_path.is_file() else []
    prior_manual: dict[str, list[dict[str, str]]] = {}
    for row in prior_rows:
        if row.get("split", "").casefold() == "train" and row.get("annotation_status") == "completed" and row.get("annotator") == "manual":
            prior_manual.setdefault(row["match_id"], []).append(row)

    annotations: list[dict[str, str]] = []
    event_manifest: list[dict[str, str]] = []
    generated = reused = 0
    step = 1.0 / SAMPLE_FPS
    offsets = [-SECONDS_BEFORE + i * step for i in range(round((SECONDS_BEFORE + SECONDS_AFTER) * SAMPLE_FPS) + 1)]
    event_center_index = min(range(len(offsets)), key=lambda i: abs(offsets[i]))

    for event in events:
        match = event["match_id"]
        filename = video_name(match)
        video_path = args.videos_dir / filename
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"TRAIN video not found: {video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or frame_count <= 0:
                raise ValueError(f"Invalid video metadata: {video_path}")
            duration = (frame_count - 1) / fps
            center_seconds = float(event["timestamp_seconds"])
            sequence_id = event["event_id"]
            event_manifest.append({
                "sequence_id": sequence_id,
                "event_role": event["event_role"],
                "pair_id": event.get("pair_id", ""),
                "matched_positive_id": event.get("matched_positive_id", ""),
                "match_id": match,
                "video_filename": filename,
                "event_timestamp_seconds": f"{center_seconds:.6f}",
                "event_timestamp_hhmmss": format_time(center_seconds),
                "event_source": event["source"],
                "source_batch": event.get("source_batch", ""),
                "window_start_seconds": f"{center_seconds - SECONDS_BEFORE:.6f}",
                "window_end_seconds": f"{center_seconds + SECONDS_AFTER:.6f}",
                "sample_count": str(len(offsets)),
                "annotation_status": "PENDING_BALL_REVIEW",
            })
            for sequence_frame, offset in enumerate(offsets, start=1):
                requested_time = center_seconds + offset
                actual_request = min(max(requested_time, 0.0), duration)
                frame_index = min(frame_count - 1, max(0, round(actual_request * fps)))
                actual_time = frame_index / fps
                sample_id = f"{sequence_id}_frame_{sequence_frame:03d}"
                image_path = args.frames_dir / match / f"{sample_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                relative_image_path = image_path.resolve().relative_to(ROOT).as_posix()
                saved = previous.get(sample_id, {})

                reused_row = None
                if not saved.get("annotation_status"):
                    for old in prior_manual.get(match, []):
                        if abs(float(old["timestamp_seconds"]) - actual_time) <= REUSE_TOLERANCE_SECONDS:
                            reused_row = old
                            break

                if not image_path.is_file():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, image = capture.read()
                    if not ok or image is None:
                        raise RuntimeError(f"Could not decode {video_path} at frame {frame_index}")
                    if not cv2.imwrite(str(image_path), image):
                        raise RuntimeError(f"Could not save frame {image_path}")
                    generated += 1

                if reused_row:
                    reused += 1
                status = saved.get("annotation_status", "") or ("completed" if reused_row else "pending")
                source = saved.get("annotation_source", "") or (
                    f"REUSED_EXACT_TRAIN_MANUAL_FRAME:{reused_row.get('event_id', '')}" if reused_row else ""
                )
                annotation = {field: "" for field in FIELDS}
                annotation.update({
                    "sequence_id": sequence_id,
                    "sequence_frame": str(sequence_frame),
                    "sample_id": sample_id,
                    "event_id": sequence_id,
                    "event_role": event["event_role"],
                    "pair_id": event.get("pair_id", ""),
                    "matched_positive_id": event.get("matched_positive_id", ""),
                    "is_event_frame": str(sequence_frame - 1 == event_center_index),
                    "match_id": match,
                    "video_filename": filename,
                    "capture_domain": "Veo/derivados",
                    "split": "train",
                    "timestamp_seconds": f"{actual_time:.6f}",
                    "timestamp_hhmmss": format_time(actual_time),
                    "image_path": relative_image_path,
                    "ball_visibility": saved.get("ball_visibility", reused_row.get("ball_visibility", "") if reused_row else ""),
                    "ball_of_play_box_json": saved.get("ball_of_play_box_json", reused_row.get("ball_of_play_box_json", "") if reused_row else ""),
                    "distractor_boxes_json": saved.get("distractor_boxes_json", reused_row.get("distractor_boxes_json", "[]") if reused_row else "[]"),
                    "goal_region_box_json": saved.get("goal_region_box_json", reused_row.get("goal_region_box_json", "") if reused_row else ""),
                    "goal_region_points_json": saved.get("goal_region_points_json", reused_row.get("goal_region_points_json", "") if reused_row else ""),
                    "goal_region_status": saved.get("goal_region_status", reused_row.get("goal_region_status", "") if reused_row else ""),
                    "goal_volume_points_json": saved.get("goal_volume_points_json", reused_row.get("goal_volume_points_json", "") if reused_row else ""),
                    "goal_volume_status": saved.get("goal_volume_status", reused_row.get("goal_volume_status", "") if reused_row else ""),
                    "annotation_status": status,
                    "annotator": saved.get("annotator", reused_row.get("annotator", "manual") if reused_row else ""),
                    "annotation_source": source,
                    "notes": (
                        "Confirmed TRAIN catalog goal; ball and event-frame goal geometry need manual review."
                        if event["event_role"] == "CATALOGED_GOAL"
                        else "Human-confirmed NO_GOAL control; do not infer ball visibility from event class."
                    ),
                })
                annotations.append(annotation)
        finally:
            capture.release()

    write_csv(args.manifest, event_manifest, list(event_manifest[0]))
    write_csv(args.output, annotations, FIELDS)
    protocol = {
        "experiment_id": "ORACLE-TRACKER-RETEST-EXP-0001",
        "scope": "TRAIN only; five fixed detector-holdout matches; no VALIDATION/TEST",
        "gate_decision_path": str(gate_path.relative_to(ROOT)),
        "gate_decision_sha256": sha256(gate_path),
        "gate_decision": gate["gate"]["decision"],
        "inputs": {
            "goal_catalog": str(GOALS.relative_to(ROOT)),
            "goal_catalog_sha256": source_hashes["goal_catalog"],
            "confirmed_control_candidates": str(CONTROLS.relative_to(ROOT)),
            "confirmed_control_candidates_sha256": source_hashes["confirmed_controls"],
            "detector_holdout_matches": sorted({event["match_id"] for event in events}),
        },
        "sampling": {
            "seconds_before": SECONDS_BEFORE,
            "seconds_after": SECONDS_AFTER,
            "sample_fps": SAMPLE_FPS,
            "frames_per_sequence": len(offsets),
            "event_center_frame_1_based": event_center_index + 1,
            "primary_scoring_frames_1_based": [16, 26],
            "primary_scoring_seconds_relative_to_event": [-1.0, 1.0],
            "rule": "Same window/sampling as the oracle tracker pilot; score its central 11 frames for comparable metrics.",
        },
        "tracker": {
            "script": "experiments/ball/run_ball_multitrack_pilot.py",
            "configuration": {"confidence": 0.01, "max_candidates": 10, "imgsz": 960, "device": "cpu", "verifier": None},
            "algorithm": "Existing Viterbi transition model, unchanged; no threshold or coefficient tuning.",
        },
        "event_counts": {
            "catalog_goals": sum(event["event_role"] == "CATALOGED_GOAL" for event in events),
            "human_confirmed_no_goal_controls": sum(event["event_role"] == "CONFIRMED_NO_GOAL" for event in events),
            "one_to_one_pairs_same_match": len({event.get("pair_id") for event in events if event.get("pair_id")}),
            "matches_with_confirmed_controls": sorted({event["match_id"] for event in events if event["event_role"] == "CONFIRMED_NO_GOAL"}),
            "matches_without_confirmed_control": sorted({event["match_id"] for event in events if event["event_role"] == "CATALOGED_GOAL"} - {event["match_id"] for event in events if event["event_role"] == "CONFIRMED_NO_GOAL"}),
            "sequences": len(events),
            "frames": len(annotations),
            "ball_frame_rows_reused_from_exact_manual_train_samples": reused,
            "frames_newly_extracted": generated,
        },
        "analysis_limits": [
            "The detector holdout passed its engineering gate by one additional positive frame (72/104 versus 71/104); this retest is conditional TRAIN engineering evidence, not detector promotion.",
            "There are six confirmed NO_GOAL controls across four matches; match_27 has no confirmed control. Pending labels are excluded.",
            "The full event set includes all 29 cataloged goals plus all six eligible human-confirmed controls; positive-only coverage in match_27 is descriptive.",
            "Do not tune or alter V4 from this retest; TEST remains closed.",
        ],
    }
    PROTOCOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_OUT.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"events={len(events)} sequences={len(event_manifest)} frames={len(annotations)} new_images={generated} reusable_annotations={reused}")
    print(f"manifest={args.manifest}\nannotations={args.output}\nprotocol={PROTOCOL_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
