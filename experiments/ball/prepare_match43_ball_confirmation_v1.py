"""Prepare a blinded, pre-inference BALL annotation batch for external Match 43."""
from __future__ import annotations

import csv
import json
import random
from datetime import timedelta
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "data/Videos_Edited/Match_43.mp4"
OUT = ROOT / "experiments/ball/BALL-HIGH-RES-CONFIRMATION-EXP-0001-MATCH43"
FRAMES = OUT / "frames_v1"
GOAL_TIMES = (3422.0, 3546.0, 3841.0, 3930.0)
EVENT_OFFSETS = tuple(-20.0 + 0.5 * i for i in range(80))
CONTROL_COUNT = 80
SEED = 430926


def allowed_intervals(duration: float, exclusion_radius: float = 45.0):
    blocked = sorted((max(0.0, t-exclusion_radius), min(duration, t+exclusion_radius)) for t in GOAL_TIMES)
    intervals = []
    cursor = 0.0
    for start, end in blocked:
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        intervals.append((cursor, duration))
    return intervals


def stratified_controls(duration: float, rng: random.Random) -> list[float]:
    intervals = allowed_intervals(duration)
    total = sum(end-start for start, end in intervals)
    cuts = [total * i / CONTROL_COUNT for i in range(CONTROL_COUNT+1)]
    controls = []
    for i in range(CONTROL_COUNT):
        left, right = cuts[i], cuts[i+1]
        u = rng.uniform(left, right)
        offset = 0.0
        for start, end in intervals:
            length = end-start
            if u <= offset + length:
                controls.append(start + (u-offset))
                break
            offset += length
    if len(controls) != CONTROL_COUNT:
        raise RuntimeError("Could not allocate the stratified background controls")
    return controls


def hhmmss(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing batch: {OUT}")
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    if fps <= 0 or max(GOAL_TIMES) + max(EVENT_OFFSETS) >= duration:
        cap.release()
        raise ValueError("Invalid video metadata or goal timestamp beyond video duration")

    rng = random.Random(SEED)
    records = []
    for goal_index, goal_time in enumerate(GOAL_TIMES, 1):
        for frame_offset, offset in enumerate(EVENT_OFFSETS, 1):
            records.append({
                "sample_id": f"match43_goal{goal_index}_{frame_offset:03d}",
                "match_id": "match_43", "video_filename": VIDEO.name,
                "split": "external_confirmation", "source_stratum": "goal_window",
                "source_id": f"match43_goal_{goal_index}",
                "sequence_id": f"match43_goal_{goal_index}_window",
                "sequence_frame_index": str(frame_offset),
                "reference_timestamp_seconds": f"{goal_time:.3f}",
                "sample_offset_seconds": f"{offset:.3f}",
                "timestamp_seconds": f"{goal_time+offset:.3f}",
                "fps": f"{fps:.8f}", "video_duration_seconds": f"{duration:.3f}",
                "selection_note": "Blind sample: fixed 0.5 s cadence from -20.0 to +19.5 s around catalogued goal; no detector output used.",
            })
    for i, timestamp in enumerate(stratified_controls(duration, rng), 1):
        records.append({
            "sample_id": f"match43_control_{i:03d}",
            "match_id": "match_43", "video_filename": VIDEO.name,
            "split": "external_confirmation", "source_stratum": "background_control",
            "source_id": f"match43_background_bin_{i:03d}", "sequence_id": "",
            "sequence_frame_index": "", "reference_timestamp_seconds": "",
            "sample_offset_seconds": "", "timestamp_seconds": f"{timestamp:.3f}",
            "fps": f"{fps:.8f}", "video_duration_seconds": f"{duration:.3f}",
            "selection_note": "Blind control: one seeded sample per equal-duration bin of footage at least 45 s from every catalogued goal; no detector output used.",
        })

    for row in records:
        row["frame_index"] = str(round(float(row["timestamp_seconds"]) * fps))
        row["timestamp_hhmmss"] = hhmmss(float(row["timestamp_seconds"]))
        row["image_path"] = str(FRAMES / f"{row['sample_id']}.jpg")
        row["ball_visibility"] = ""
        row["ball_x1"] = row["ball_y1"] = row["ball_x2"] = row["ball_y2"] = ""
        row["annotation_status"] = "pending"
        row["annotator"] = ""
        row["notes"] = ""

    records.sort(key=lambda r: int(r["frame_index"]))
    if len(records) != 400 or len({r["frame_index"] for r in records}) != 400:
        cap.release()
        raise ValueError("Expected 400 unique frame indices")

    OUT.mkdir(parents=True)
    FRAMES.mkdir(parents=True)
    manifest_path = OUT / "match43_ball_annotation_manifest_v1.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    decoded = 0
    for row in records:
        frame_index = int(row["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Failed decoding {row['sample_id']} at frame {frame_index}")
        if not cv2.imwrite(row["image_path"], frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            cap.release()
            raise RuntimeError(f"Failed writing {row['image_path']}")
        decoded += 1
        if decoded % 50 == 0:
            print(f"Extracted {decoded}/400 frames", flush=True)
    cap.release()

    metadata = {
        "status": "prepared_for_blind_manual_annotation",
        "match_id": "match_43", "video": str(VIDEO),
        "video_size_bytes": VIDEO.stat().st_size,
        "fps": fps, "frame_count": frame_count, "width": width, "height": height,
        "duration_seconds": duration, "catalogued_goal_times_seconds": GOAL_TIMES,
        "sample_count": len(records), "goal_window_frames": 320,
        "background_control_frames": CONTROL_COUNT, "seed": SEED,
        "test_accessed": False,
        "warning": "Annotate every frame before loading/running any detector. Match 43 is an external match-level confirmation relative to existing TRAIN/VALIDATION, conditional on confirming this video was not used elsewhere.",
    }
    (OUT / "preparation_metadata_v1.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
