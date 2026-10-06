"""Run the frozen Viterbi association over label-blind tiled proposal lists."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/blind_tracker_inputs_full_v1.csv"
DEFAULT_EXPERIMENT = ROOT / "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001"
PREDICTIONS = DEFAULT_EXPERIMENT / "tracker_candidates_blind_v1.csv"
CHECKPOINTS = {
    "baseline": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "oversampled": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
}
FRAME_FIELDS = ("checkpoint", "sequence_id", "sequence_frame", "sample_id", "candidate_count", "selected_candidate_rank", "selected_confidence", "selected_box_json", "candidate_boxes_json", "image_width", "image_height", "track_prediction_present")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def transition_penalty(previous, current, diagonal: float) -> float:
    p, c = center(previous), center(current)
    distance = math.hypot(p[0] - c[0], p[1] - c[1]) / diagonal
    previous_area = max(1.0, (previous[2] - previous[0]) * (previous[3] - previous[1]))
    current_area = max(1.0, (current[2] - current[0]) * (current[3] - current[1]))
    return 4.0 * distance + 0.25 * abs(math.log(current_area / previous_area))


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=PREDICTIONS)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    freeze_path = args.experiment_dir / "tiled_tracker_run_freeze_v1.json"
    if freeze_path.exists():
        raise FileExistsError(f"Prediction freeze already exists: {freeze_path}")
    inputs = read_csv(args.input)
    candidates = read_csv(args.candidates)
    if len(inputs) != 1435 or len({row["sequence_id"] for row in inputs}) != 35:
        raise ValueError("Expected 35 blind sequences and 1,435 frames")
    if not candidates:
        raise ValueError("No tiled candidate predictions")
    candidate_by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for source in candidates:
        box = tuple(float(source[key]) for key in ("x1", "y1", "x2", "y2"))
        candidate_by_key[(source["checkpoint"], source["sample_id"])].append({
            "box": box, "confidence": float(source["confidence"]), "rank": int(source["rank"]),
        })
    if {source["checkpoint"] for source in candidates} != set(CHECKPOINTS):
        raise ValueError("Both frozen checkpoints must be represented in the proposal file")
    progress_path = args.experiment_dir / "candidate_generation_progress_v1.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected_keys = {f"{model}|{row['sample_id']}" for model in CHECKPOINTS for row in inputs}
    if set(progress.get("completed", [])) != expected_keys:
        raise ValueError("Tiled proposal generator has not completed every checkpoint/frame pair")
    proposal_keys = [(row["checkpoint"], row["sample_id"], row["rank"]) for row in candidates]
    if len(proposal_keys) != len(set(proposal_keys)):
        raise ValueError("Duplicate checkpoint/sample/rank rows in the tiled proposal CSV")

    dim_by_sample: dict[str, tuple[int, int]] = {}
    for row in inputs:
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        dim_by_sample[row["sample_id"]] = (int(image.shape[1]), int(image.shape[0]))

    frame_output: list[dict[str, object]] = []
    sequence_output: list[dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in inputs:
            by_sequence[row["sequence_id"]].append(row)
        for sequence_id, sequence_rows in by_sequence.items():
            sequence_rows.sort(key=lambda row: int(row["sequence_frame"]))
            candidates_by_frame = [candidate_by_key[(checkpoint, row["sample_id"])] for row in sequence_rows]
            scores: list[list[float]] = []
            backpointers: list[list[int]] = []
            for frame_index, candidate_rows in enumerate(candidates_by_frame):
                if not candidate_rows:
                    scores.append([])
                    backpointers.append([])
                    continue
                if frame_index == 0 or not scores[-1]:
                    scores.append([float(row["confidence"]) for row in candidate_rows])
                    backpointers.append([-1] * len(candidate_rows))
                    continue
                width, height = dim_by_sample[sequence_rows[frame_index]["sample_id"]]
                diagonal = math.hypot(width, height)
                current_scores, current_backpointers = [], []
                for candidate in candidate_rows:
                    options = [
                        previous_score - transition_penalty(candidates_by_frame[frame_index - 1][previous_index]["box"], candidate["box"], diagonal)
                        for previous_index, previous_score in enumerate(scores[-1])
                    ]
                    previous_index = max(range(len(options)), key=options.__getitem__)
                    current_scores.append(float(candidate["confidence"]) + options[previous_index])
                    current_backpointers.append(previous_index)
                scores.append(current_scores)
                backpointers.append(current_backpointers)

            available = [index for index, values in enumerate(scores) if values]
            selected: dict[int, int] = {}
            if available:
                current = available[-1]
                candidate_index = max(range(len(scores[current])), key=scores[current].__getitem__)
                for frame_index in range(current, -1, -1):
                    if candidate_index < 0 or not candidates_by_frame[frame_index]:
                        break
                    selected[frame_index] = candidate_index
                    candidate_index = backpointers[frame_index][candidate_index]

            selected_flags: list[bool] = []
            confidences: list[float] = []
            for frame_index, row in enumerate(sequence_rows):
                items = candidates_by_frame[frame_index]
                selected_index = selected.get(frame_index, -1)
                selected_item = items[selected_index] if selected_index >= 0 else None
                selected_flags.append(selected_item is not None)
                if selected_item:
                    confidences.append(float(selected_item["confidence"]))
                width, height = dim_by_sample[row["sample_id"]]
                frame_output.append({
                    "checkpoint": checkpoint, "sequence_id": sequence_id,
                    "sequence_frame": row["sequence_frame"], "sample_id": row["sample_id"],
                    "candidate_count": len(items),
                    "selected_candidate_rank": selected_item["rank"] if selected_item else 0,
                    "selected_confidence": f"{float(selected_item['confidence']):.8f}" if selected_item else "",
                    "selected_box_json": json.dumps(selected_item["box"]) if selected_item else "",
                    "candidate_boxes_json": json.dumps([
                        {"box": item["box"], "confidence": item["confidence"], "rank": item["rank"]}
                        for item in items
                    ], separators=(",", ":")),
                    "image_width": width, "image_height": height,
                    "track_prediction_present": selected_item is not None,
                })
            sequence_output.append({
                "checkpoint": checkpoint, "sequence_id": sequence_id,
                "frames": len(sequence_rows), "frames_with_candidates": sum(bool(values) for values in candidates_by_frame),
                "frames_selected_by_viterbi": sum(selected_flags),
                "selected_track_coverage": f"{sum(selected_flags) / len(sequence_rows):.6f}",
                "longest_selected_run": longest_run(selected_flags),
                "mean_selected_confidence": f"{sum(confidences) / len(confidences):.8f}" if confidences else "",
            })
            print(f"Viterbi prediction-only: {checkpoint} {sequence_id}", flush=True)

    frame_path = args.experiment_dir / "tiled_tracker_frame_predictions_v1.csv"
    sequence_path = args.experiment_dir / "tiled_tracker_sequence_predictions_v1.csv"
    write_csv(frame_path, frame_output)
    write_csv(sequence_path, sequence_output)
    freeze = {
        "experiment_id": args.experiment_dir.name, "prediction_only": True,
        "labels_included": False, "scope": "TRAIN only; TEST remains closed; no V4 promotion",
        "input": {"path": str(args.input.relative_to(ROOT)), "sha256": sha256(args.input), "rows": len(inputs)},
        "candidate_predictions": {"path": str(args.candidates.relative_to(ROOT)), "sha256": sha256(args.candidates), "rows": len(candidates)},
        "output": {
            "frame_predictions": {"path": str(frame_path.relative_to(ROOT)), "sha256": sha256(frame_path), "rows": len(frame_output)},
            "sequence_predictions": {"path": str(sequence_path.relative_to(ROOT)), "sha256": sha256(sequence_path), "rows": len(sequence_output)},
        },
        "checkpoint_hashes": {name: sha256(path) for name, path in CHECKPOINTS.items()},
        "tracker": {"transition_penalty": "4*normalized center distance + 0.25*abs(log area ratio)", "score": "sum detector confidences minus transition penalties", "confidence": 0.01, "max_candidates": 10, "imgsz": 960, "tiles": "2x2, overlap 0.20", "cross_view_nms_iou": 0.75},
    }
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN_PREDICTION_ONLY", "frame_rows": len(frame_output), "sequence_rows": len(sequence_output), "freeze": str(freeze_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
