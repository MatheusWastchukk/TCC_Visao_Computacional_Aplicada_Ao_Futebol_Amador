"""Replay an anchored nearest-trajectory tracker over frozen blind Top-10 detections."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
TRACK = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001"
PROTOCOL = EXP / "protocol_v1.json"
RUN_FREEZE = TRACK / "blind_tracker_run_freeze_v1.json"
RUNS = {
    ("center", "baseline"): TRACK / "center_baseline",
    ("center", "oversampled"): TRACK / "center_oversampled",
    ("full", "baseline"): TRACK / "full_baseline",
    ("full", "oversampled"): TRACK / "full_oversampled",
}
INPUTS = {
    "center": TRACK / "blind_tracker_inputs_center_v1.csv",
    "full": TRACK / "blind_tracker_inputs_full_v1.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def run_sequence(rows: list[dict[str, str]], diagonal_by_sample: dict[str, float], seed_window: int, gate_fraction: float, max_gap: int) -> list[dict[str, object]]:
    candidates_by_frame = [json.loads(row["candidate_boxes_json"]) for row in rows]
    seed_choices = [
        (float(candidate["confidence"]), -i, -j, i, j)
        for i, candidates in enumerate(candidates_by_frame[:seed_window])
        for j, candidate in enumerate(candidates)
    ]
    seed_index = max(seed_choices)[3] if seed_choices else None
    selected: dict[int, int] = {}
    predicted_by_frame: dict[int, tuple[float, float]] = {}
    distance_by_frame: dict[int, float] = {}
    gate_by_frame: dict[int, float] = {}
    status_by_frame = {i: "BEFORE_SEED" for i in range(len(rows))}
    seed_sample_id = ""
    if seed_index is not None:
        seed_candidate = max(
            range(len(candidates_by_frame[seed_index])),
            key=lambda j: (float(candidates_by_frame[seed_index][j]["confidence"]), -j),
        )
        selected[seed_index] = seed_candidate
        seed_sample_id = rows[seed_index]["sample_id"]
        status_by_frame[seed_index] = "SEED_HIGHEST_CONFIDENCE"
        history: list[tuple[int, list[float]]] = [(seed_index, list(candidates_by_frame[seed_index][seed_candidate]["box"]))]

        for index in range(seed_index + 1, len(rows)):
            gap = index - history[-1][0]
            if gap > max_gap:
                for lost_index in range(index, len(rows)):
                    status_by_frame[lost_index] = "TRACK_LOST_MAX_GAP"
                break
            last_index, last_box = history[-1]
            last_center = box_center(last_box)
            if len(history) >= 2:
                previous_index, previous_box = history[-2]
                previous_center = box_center(previous_box)
                elapsed = last_index - previous_index
                vx = (last_center[0] - previous_center[0]) / elapsed if elapsed else 0.0
                vy = (last_center[1] - previous_center[1]) / elapsed if elapsed else 0.0
                predicted = (last_center[0] + vx * gap, last_center[1] + vy * gap)
            else:
                predicted = last_center
            predicted_by_frame[index] = predicted
            diagonal = diagonal_by_sample[rows[index]["sample_id"]]
            gate_px = gate_fraction * diagonal * gap
            gate_by_frame[index] = gate_px
            frame_candidates = candidates_by_frame[index]
            eligible: list[tuple[float, float, int]] = []
            for candidate_index, candidate in enumerate(frame_candidates):
                cx, cy = box_center(candidate["box"])
                distance = math.hypot(cx - predicted[0], cy - predicted[1])
                if distance <= gate_px:
                    eligible.append((distance, -float(candidate["confidence"]), candidate_index))
            if eligible:
                distance, _, candidate_index = min(eligible)
                selected[index] = candidate_index
                distance_by_frame[index] = distance
                status_by_frame[index] = "TRACKED_NEAREST_TO_TRAJECTORY"
                history.append((index, list(frame_candidates[candidate_index]["box"])))
            else:
                status_by_frame[index] = "GAP_NO_CANDIDATE_WITHIN_GATE"
    else:
        status_by_frame = {i: "NO_SEED_CANDIDATES" for i in range(len(rows))}

    out = []
    for index, row in enumerate(rows):
        candidates = candidates_by_frame[index]
        choice = selected.get(index)
        selected_item = candidates[choice] if choice is not None else None
        predicted = predicted_by_frame.get(index)
        diagonal = diagonal_by_sample[row["sample_id"]]
        out.append({
            "window": "",
            "detector": "",
            "sequence_id": row["sequence_id"],
            "sequence_frame": row["sequence_frame"],
            "sample_id": row["sample_id"],
            "seed_sample_id": seed_sample_id,
            "anchor_status": status_by_frame[index],
            "candidate_count": len(candidates),
            "selected_candidate_rank": choice + 1 if choice is not None else 0,
            "selected_confidence": f"{float(selected_item['confidence']):.8f}" if selected_item else "",
            "selected_box_json": json.dumps(selected_item["box"]) if selected_item else "",
            "predicted_center_json": json.dumps(predicted) if predicted else "",
            "distance_to_trajectory_px": f"{distance_by_frame[index]:.6f}" if index in distance_by_frame else "",
            "distance_to_trajectory_fraction": f"{distance_by_frame[index] / diagonal:.8f}" if index in distance_by_frame else "",
            "gate_radius_px": f"{gate_by_frame[index]:.6f}" if index in gate_by_frame else "",
        })
    return out


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_freeze = json.loads(RUN_FREEZE.read_text(encoding="utf-8"))
    if not run_freeze.get("prediction_only"):
        raise ValueError("Frozen label-blind tracker predictions are required.")
    freeze_path = EXP / "blind_replay_freeze_v1.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to replace existing replay freeze: {freeze_path}")
    seed_window = int(protocol["algorithm"]["seed_window_frames"])
    gate_fraction = float(protocol["algorithm"]["hard_gate_fraction_of_image_diagonal_per_frame"])
    max_gap = int(protocol["algorithm"]["max_gap_frames_before_track_lost"])
    output_hashes: dict[str, dict[str, object]] = {}

    image_paths: dict[str, str] = {}
    diagonal_by_sample: dict[str, float] = {}
    for window, path in INPUTS.items():
        for row in read_csv(path):
            image_paths[row["sample_id"]] = row["image_path"]
    for sample_id, relative in image_paths.items():
        image_path = Path(relative) if Path(relative).is_absolute() else ROOT / relative
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        diagonal_by_sample[sample_id] = math.hypot(image.shape[0], image.shape[1])

    for (window, detector), source_dir in RUNS.items():
        run_name = f"{window}_{detector}"
        frozen_run = run_freeze["runs"][run_name]
        source_path = source_dir / "frame_identity_results.csv"
        if sha256(source_path) != frozen_run["frame_results_sha256"]:
            raise ValueError(f"Source blind predictions changed: {run_name}")
        source_rows = read_csv(source_path)
        safe_rows = read_csv(INPUTS[window])
        if [r["sample_id"] for r in source_rows] != [r["sample_id"] for r in safe_rows]:
            raise ValueError(f"Prediction/input sample IDs differ: {run_name}")
        by_sequence: dict[str, list[dict[str, str]]] = {}
        for row in source_rows:
            if not row.get("candidate_boxes_json"):
                raise ValueError(f"Missing frozen candidates at {row['sample_id']}")
            by_sequence.setdefault(row["sequence_id"], []).append(row)
        replay_rows: list[dict[str, object]] = []
        for sequence_id in sorted(by_sequence):
            seq_rows = sorted(by_sequence[sequence_id], key=lambda r: int(r["sequence_frame"]))
            replay = run_sequence(seq_rows, diagonal_by_sample, seed_window, gate_fraction, max_gap)
            for row in replay:
                row["window"] = window
                row["detector"] = detector
            replay_rows.extend(replay)
        output_path = EXP / run_name / "anchor_trajectory_predictions_v1.csv"
        if output_path.exists():
            raise FileExistsError(f"Refusing to replace existing replay output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(replay_rows[0]))
            writer.writeheader()
            writer.writerows(replay_rows)
        output_hashes[run_name] = {
            "source_predictions_sha256": sha256(source_path),
            "anchor_predictions_path": str(output_path.relative_to(ROOT)),
            "anchor_predictions_rows": len(replay_rows),
            "anchor_predictions_sha256": sha256(output_path),
        }
        print(f"Replayed {run_name}: {len(replay_rows)} frames, {len(by_sequence)} sequences")

    freeze = {
        "experiment_id": protocol["experiment_id"],
        "scope": protocol["scope"],
        "protocol_sha256": sha256(PROTOCOL),
        "source_prediction_freeze_sha256": sha256(RUN_FREEZE),
        "tracker_configuration": protocol["algorithm"],
        "runs": output_hashes,
        "label_blind": True,
        "ground_truth_read": False,
    }
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "freeze": str(freeze_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
