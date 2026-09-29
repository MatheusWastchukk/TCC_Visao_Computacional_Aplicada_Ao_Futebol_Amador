"""Prepara amostra estratificada de 400 frames TRAIN para auditar propostas BALL."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "experiments/datasets/dataset_split_v1.csv"
GOALS = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/train_goal_pool_and_shortlist_coverage_v1.csv"
MISSES = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/manual_review_v1.csv"
GROUPS = ROOT / "results/goal_event_temporal_group_reconstruction_train_v1/temporal_group_dataset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative-count", type=int, default=112)
    parser.add_argument("--miss-frames-per-goal", type=int, default=12)
    parser.add_argument("--easy-goals", type=int, default=24)
    parser.add_argument("--frames-per-easy-goal", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    splits = {r["match_id"]: r for r in read_csv(SPLIT)}
    if any(r["split"] != "train" for r in splits.values() if r["match_id"] in {g["match_id"] for g in read_csv(GOALS)}):
        raise AssertionError("TRAIN-only catalog mismatch; TEST must remain closed.")
    goal_rows = read_csv(GOALS)
    if len(goal_rows) != 152 or any(g["split"] != "train" for g in goal_rows):
        raise ValueError("Expected 152 TRAIN catalog goals only.")
    goal_times: dict[str, list[float]] = defaultdict(list)
    for g in goal_rows:
        goal_times[g["match_id"]].append(float(g["event_timestamp_seconds"]))
    miss_rows = read_csv(MISSES)
    if len(miss_rows) != 12 or any(r.get("split", "train") != "train" for r in miss_rows):
        raise ValueError("Expected the 12 audited TRAIN misses only.")
    miss_ids = {r["goal_catalog_id"] for r in miss_rows}

    # Pick 24 easy positive controls, one per match when possible, covered by the frozen Top-60.
    easy = [g for g in goal_rows if g["goal_catalog_id"] not in miss_ids and g["is_covered_by_rrf_nms_top60"].lower() == "true"]
    easy_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for g in easy:
        easy_by_match[g["match_id"]].append(g)
    rng = random.Random(args.seed)
    for values in easy_by_match.values():
        values.sort(key=lambda r: r["goal_catalog_id"])
    selected_easy: list[dict[str, str]] = []
    for round_index in range(max(map(len, easy_by_match.values()))):
        for match in sorted(easy_by_match):
            if round_index < len(easy_by_match[match]) and len(selected_easy) < args.easy_goals:
                selected_easy.append(easy_by_match[match][round_index])
    if len(selected_easy) != args.easy_goals:
        raise ValueError(f"Could only select {len(selected_easy)} easy goals.")

    proposals: list[dict[str, object]] = []
    used_frames: set[tuple[str, int]] = set()
    metadata: dict[str, dict[str, float]] = {}

    def video_meta(match: str) -> tuple[float, float]:
        if match not in metadata:
            video_path = ROOT / "data/Videos_Edited" / splits[match]["video_filename"]
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(video_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps
            cap.release()
            metadata[match] = {"fps": fps, "duration": duration}
        return metadata[match]["fps"], metadata[match]["duration"]

    def add(match: str, video: str, center: float, offset: float, stratum: str, source_id: str,
            sequence_id: str = "", sequence_frame_index: int | str = "") -> None:
        fps, duration = video_meta(match)
        seconds = min(max(center + offset, 0.0), max(0.0, duration - 1.0 / fps))
        frame_index = round(seconds * fps)
        key = (match, frame_index)
        if key in used_frames:
            # Move by one sampled step while preserving the local event window.
            for delta in (1, -1, 2, -2, 3, -3, 4, -4):
                alternative = frame_index + delta * max(1, round(fps / 5.0))
                if alternative >= 0 and alternative < round(duration * fps) and (match, alternative) not in used_frames:
                    frame_index = alternative
                    key = (match, frame_index)
                    break
        if key in used_frames:
            raise ValueError(f"Could not make unique frame sample: {match} {seconds:.3f}")
        used_frames.add(key)
        proposals.append({"sample_id": f"ball_proposal_{len(proposals)+1:04d}", "match_id": match,
            "video_filename": video, "split": "train", "source_stratum": stratum, "source_id": source_id,
            "sequence_id": sequence_id, "sequence_frame_index": sequence_frame_index,
            "reference_timestamp_seconds": f"{center:.3f}", "sample_offset_seconds": f"{offset:.3f}",
            "timestamp_seconds": f"{frame_index/fps:.3f}", "frame_index": frame_index,
            "fps": f"{fps:.6f}", "video_duration_seconds": f"{duration:.3f}"})

    if args.miss_frames_per_goal != 12:
        raise ValueError("This version uses 12 frames per miss: four 3-frame bursts.")
    for miss in miss_rows:
        center = float(miss["goal_timestamp_seconds"])
        for burst_index, burst_center in enumerate((-20.0, -5.0, 5.0, 25.0), 1):
            sequence_id = f"{miss['goal_catalog_id']}__burst{burst_index}"
            for frame_order in range(3):
                offset = burst_center + frame_order * 0.2
                add(miss["match_id"], miss["video_filename"], center, offset, "audited_miss",
                    miss["goal_catalog_id"], sequence_id, frame_order + 1)

    easy_offsets = [-1.0, -0.8, -0.6, 1.0, 1.2, 1.4]
    if args.frames_per_easy_goal != len(easy_offsets):
        raise ValueError("This version uses exactly six offsets per easy goal.")
    for goal in selected_easy:
        for burst_index, start_offset in enumerate((-1.0, 1.0), 1):
            sequence_id = f"{goal['goal_catalog_id']}__burst{burst_index}"
            for frame_order, offset in enumerate((start_offset, start_offset + 0.2, start_offset + 0.4), 1):
                add(goal["match_id"], goal["video_filename"], float(goal["event_timestamp_seconds"]), offset,
                    "easy_goal_control", goal["goal_catalog_id"], sequence_id, frame_order)

    # Weak-negative candidate groups outside 60 s of any catalog goal; stratify by match.
    negative_rows = [r for r in read_csv(GROUPS) if int(r["weak_label"]) == 0 and
        min(abs(float(r["best_timestamp_seconds"]) - t) for t in goal_times[r["match_id"]]) > 60.0]
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in negative_rows:
        candidates[row["match_id"]].append(row)
    for values in candidates.values():
        rng.shuffle(values)
    selected_negative: list[dict[str, str]] = []
    selected_times: dict[str, list[float]] = defaultdict(list)
    # Round-robin across matches to avoid letting a few games dominate the control sample.
    cursor_by_match: dict[str, int] = defaultdict(int)
    while len(selected_negative) < args.negative_count:
        added = False
        for match in sorted(candidates):
            values = candidates[match]
            while cursor_by_match[match] < len(values):
                candidate = values[cursor_by_match[match]]
                t = float(candidate["best_timestamp_seconds"])
                cursor_by_match[match] += 1
                if all(abs(t - other) >= 30.0 for other in selected_times[match]):
                    selected_negative.append(candidate)
                    selected_times[match].append(t)
                    added = True
                    break
            if len(selected_negative) == args.negative_count:
                break
        if not added:
            raise ValueError(f"Only {len(selected_negative)} weak-negative controls could be selected.")
    for row in selected_negative:
        add(row["match_id"], splits[row["match_id"]]["video_filename"], float(row["best_timestamp_seconds"]),
            0.0, "weak_negative_control", row["group_uid"])

    if len(proposals) != 400 or len(used_frames) != 400:
        raise AssertionError(f"Expected exactly 400 unique frames; got {len(proposals)}")
    args.output.mkdir(parents=True)
    write_csv(args.output / "sample_manifest_v1.csv", proposals)
    counts = defaultdict(int)
    for row in proposals:
        counts[str(row["source_stratum"])] += 1
    summary = {"experiment_id": args.output.name, "scope": "TRAIN only; TEST closed", "sample_count": len(proposals),
        "strata": dict(counts), "unique_frames": len(used_frames), "unique_matches": len({r["match_id"] for r in proposals}),
        "sampling": {"miss_frames_per_goal": args.miss_frames_per_goal, "easy_positive_goals": len(selected_easy),
            "frames_per_easy_goal": len(easy_offsets), "weak_negative_controls": args.negative_count,
            "sequence_sampling": "4 bursts x 3 consecutive analysis frames at 5 FPS around each audited miss; 2 bursts x 3 frames for each easy goal control",
            "weak_negative_definition": "weak_label=0 and more than 60 s from every TRAIN catalog goal; not human ground truth",
            "random_seed": args.seed},
        "annotation_required": ["tight box around the real ball, or BALL_NOT_VISIBLE/UNCERTAIN", "optional scene tags"]}
    (args.output / "sample_summary_v1.json").write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(__import__("json").dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
