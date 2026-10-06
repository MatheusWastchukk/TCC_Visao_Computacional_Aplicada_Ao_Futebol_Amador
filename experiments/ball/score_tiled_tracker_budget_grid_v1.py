"""Score a label-blind Viterbi grid from the already saved Top-10 tiled proposals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001"
DEFAULT_ANNOTATIONS = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/frame_annotations_v1.csv"
CONFIDENCE_FLOORS = (0.01, 0.03, 0.05)
TOP_N_VALUES = (1, 3, 5, 10)
FRAME_PREDICTIONS = "tiled_tracker_frame_predictions_v1.csv"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_box(raw: str):
    if not raw:
        return None
    values = json.loads(raw)
    if len(values) != 4:
        raise ValueError(f"Invalid xyxy box: {values}")
    return tuple(float(value) for value in values)


def iou(a, b) -> float:
    x1, y1, x2, y2 = a
    gx1, gy1, gx2, gy2 = b
    intersection = max(0.0, min(x2, gx2) - max(x1, gx1)) * max(0.0, min(y2, gy2) - max(y1, gy1))
    area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_b = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def transition_penalty(previous, current, diagonal: float) -> float:
    previous_center = ((previous[0] + previous[2]) / 2.0, (previous[1] + previous[3]) / 2.0)
    current_center = ((current[0] + current[2]) / 2.0, (current[1] + current[3]) / 2.0)
    distance = math.hypot(previous_center[0] - current_center[0], previous_center[1] - current_center[1]) / diagonal
    previous_area = max(1.0, (previous[2] - previous[0]) * (previous[3] - previous[1]))
    current_area = max(1.0, (current[2] - current[0]) * (current[3] - current[1]))
    return 4.0 * distance + 0.25 * abs(math.log(current_area / previous_area))


def run_viterbi(sequence_rows: list[dict[str, str]], confidence_floor: float, top_n: int) -> list[dict[str, object]]:
    candidates_by_frame: list[list[dict[str, object]]] = []
    for row in sequence_rows:
        available = json.loads(row.get("candidate_boxes_json", "[]") or "[]")
        available = [item for item in available if float(item["confidence"]) >= confidence_floor]
        available.sort(key=lambda item: (-float(item["confidence"]), int(item.get("rank", 0))))
        candidates_by_frame.append(available[:top_n])

    scores: list[list[float]] = []
    backpointers: list[list[int]] = []
    for frame_index, candidates in enumerate(candidates_by_frame):
        if not candidates:
            scores.append([])
            backpointers.append([])
            continue
        if frame_index == 0 or not scores[-1]:
            scores.append([float(item["confidence"]) for item in candidates])
            backpointers.append([-1] * len(candidates))
            continue
        width, height = int(sequence_rows[frame_index]["image_width"]), int(sequence_rows[frame_index]["image_height"])
        diagonal = math.hypot(width, height)
        current_scores: list[float] = []
        current_backpointers: list[int] = []
        for candidate in candidates:
            box = tuple(float(value) for value in candidate["box"])
            options = [
                previous_score - transition_penalty(
                    tuple(float(value) for value in candidates_by_frame[frame_index - 1][previous_index]["box"]),
                    box,
                    diagonal,
                )
                for previous_index, previous_score in enumerate(scores[-1])
            ]
            previous_index = max(range(len(options)), key=options.__getitem__)
            current_scores.append(float(candidate["confidence"]) + options[previous_index])
            current_backpointers.append(previous_index)
        scores.append(current_scores)
        backpointers.append(current_backpointers)

    available_frames = [index for index, values in enumerate(scores) if values]
    selected: dict[int, dict[str, object]] = {}
    if available_frames:
        current = available_frames[-1]
        candidate_index = max(range(len(scores[current])), key=scores[current].__getitem__)
        for frame_index in range(current, -1, -1):
            if candidate_index < 0 or not candidates_by_frame[frame_index]:
                break
            selected[frame_index] = candidates_by_frame[frame_index][candidate_index]
            candidate_index = backpointers[frame_index][candidate_index]

    results: list[dict[str, object]] = []
    for index, row in enumerate(sequence_rows):
        annotation = row["_annotation"]
        target_visible = annotation["ball_visibility"] in {"VISIBLE", "PARTIAL"}
        target = parse_box(annotation.get("ball_of_play_box_json", "")) if target_visible else None
        distractors = [
            tuple(float(value) for value in box)
            for box in json.loads(annotation.get("distractor_boxes_json", "[]") or "[]")
        ]
        candidate_boxes = [tuple(float(value) for value in item["box"]) for item in candidates_by_frame[index]]
        candidate_rank = next((rank for rank, box in enumerate(candidate_boxes, 1) if target and iou(box, target) >= 0.5), 0)
        chosen = selected.get(index)
        chosen_box = tuple(float(value) for value in chosen["box"]) if chosen else None
        target_iou = iou(chosen_box, target) if chosen_box and target else 0.0
        distractor_iou = max((iou(chosen_box, box) for box in distractors), default=0.0) if chosen_box else 0.0
        results.append({
            "sequence_id": row["sequence_id"],
            "sequence_frame": int(row["sequence_frame"]),
            "sample_id": row["sample_id"],
            "event_role": annotation["event_role"],
            "pair_id": annotation["pair_id"],
            "target_annotated": target is not None,
            "candidate_hit": candidate_rank > 0,
            "identity_correct": target_iou >= 0.5,
            "wrong_ball_switch": distractor_iou >= 0.5,
            "candidate_rank": candidate_rank,
            "candidate_count": len(candidate_boxes),
            "selected_confidence": float(chosen["confidence"]) if chosen else "",
            "selected_target_iou": target_iou,
        })
    return results


def aggregate(rows: list[dict[str, object]], scope: str, role: str = "ALL") -> dict[str, object]:
    selected = [row for row in rows if row["target_annotated"] and (role == "ALL" or row["event_role"] == role)]
    if not selected:
        return {"scope": scope, "event_role": role, "target_frames": 0}
    count = len(selected)
    return {
        "scope": scope,
        "event_role": role,
        "target_frames": count,
        "candidate_hits": sum(bool(row["candidate_hit"]) for row in selected),
        "candidate_recall": f"{sum(bool(row['candidate_hit']) for row in selected) / count:.6f}",
        "identity_correct_frames": sum(bool(row["identity_correct"]) for row in selected),
        "identity_accuracy": f"{sum(bool(row['identity_correct']) for row in selected) / count:.6f}",
        "wrong_ball_switch_frames": sum(bool(row["wrong_ball_switch"]) for row in selected),
        "wrong_ball_switch_rate": f"{sum(bool(row['wrong_ball_switch']) for row in selected) / count:.6f}",
        "mean_proposals_per_target_frame": f"{sum(int(row['candidate_count']) for row in selected) / count:.6f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.experiment_dir.is_absolute():
        args.experiment_dir = ROOT / args.experiment_dir
    args.experiment_dir = args.experiment_dir.resolve()
    if args.output is None:
        args.output = args.experiment_dir / "budget_grid_v1"
    elif not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")

    freeze_path = args.experiment_dir / "tiled_tracker_run_freeze_v1.json"
    frame_path = args.experiment_dir / FRAME_PREDICTIONS
    if not freeze_path.is_file() or not frame_path.is_file():
        raise FileNotFoundError("Frozen label-blind tiled frame predictions are required")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("prediction_only") or freeze.get("labels_included"):
        raise AssertionError("Tiled prediction input is not label-blind")
    if freeze.get("tracker", {}).get("confidence") != 0.01 or freeze.get("tracker", {}).get("max_candidates") != 10:
        raise ValueError("This budget grid is registered only for the saved conf=0.01, Top-10 proposal set")

    annotations = read_csv(args.annotations)
    if len(annotations) != 1435 or any(row["split"].lower() != "train" for row in annotations):
        raise ValueError("Expected exactly 1,435 TRAIN annotation rows")
    annotation_by_id = {row["sample_id"]: row for row in annotations}
    if len(annotation_by_id) != 1435:
        raise ValueError("Duplicate sample_id in annotations")
    frame_predictions = read_csv(frame_path)
    if len(frame_predictions) != 2870 or len({(row["checkpoint"], row["sample_id"]) for row in frame_predictions}) != 2870:
        raise ValueError("Expected unique label-blind predictions for 2 checkpoints x 1,435 frames")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in frame_predictions:
        if row["sample_id"] not in annotation_by_id:
            raise ValueError(f"Prediction has no annotation row: {row['sample_id']}")
        grouped[(row["checkpoint"], row["sequence_id"])].append({
            **row,
            "_annotation": annotation_by_id[row["sample_id"]],
        })
    if {key[0] for key in grouped} != {"baseline", "oversampled"} or len({key[1] for key in grouped}) != 35:
        raise ValueError("Unexpected frozen checkpoint or TRAIN sequence set")

    summary_rows: list[dict[str, object]] = []
    sequence_rows_out: list[dict[str, object]] = []
    scored_by_configuration: dict[tuple[float, int, str], list[dict[str, object]]] = defaultdict(list)
    for confidence_floor in CONFIDENCE_FLOORS:
        for top_n in TOP_N_VALUES:
            for (checkpoint, sequence_id), rows in sorted(grouped.items()):
                rows.sort(key=lambda row: int(row["sequence_frame"]))
                scored = run_viterbi(rows, confidence_floor, top_n)
                annotation_role = rows[0]["_annotation"]["event_role"]
                sequence_summary = aggregate(scored, "full")
                sequence_rows_out.append({
                    "checkpoint": checkpoint,
                    "confidence_floor": f"{confidence_floor:.3f}",
                    "top_n": top_n,
                    "sequence_id": sequence_id,
                    "event_role": annotation_role,
                    **sequence_summary,
                })
                scored_by_configuration[(confidence_floor, top_n, checkpoint)].extend(scored)

    for (confidence_floor, top_n, checkpoint), scored in sorted(scored_by_configuration.items()):
        scopes = {
            "full": scored,
            "center_16_26": [row for row in scored if 16 <= int(row["sequence_frame"]) <= 26],
        }
        for scope, scope_rows in scopes.items():
            for role in ("ALL", "CATALOGED_GOAL", "CONFIRMED_NO_GOAL"):
                summary_rows.append({
                    "checkpoint": checkpoint,
                    "confidence_floor": f"{confidence_floor:.3f}",
                    "top_n": top_n,
                    **aggregate(scope_rows, scope, role),
                })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "tiled_tracker_budget_grid_summary_v1.csv", summary_rows)
    write_csv(args.output / "tiled_tracker_budget_grid_by_sequence_v1.csv", sequence_rows_out)
    manifest = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; label-blind tracker rerun using frozen human annotations only after prediction recomputation; VALIDATION and TEST not accessed",
        "prediction_freeze_sha256": sha256(freeze_path),
        "frame_prediction_sha256": sha256(frame_path),
        "annotation_sha256": sha256(args.annotations),
        "candidate_generation_limits": {
            "source_confidence_floor": 0.01,
            "source_max_candidates_per_frame": 10,
            "source_checkpoint_hashes": freeze["checkpoint_hashes"],
        },
        "grid": {
            "confidence_floors": list(CONFIDENCE_FLOORS),
            "top_n": list(TOP_N_VALUES),
            "viterbi_transition_penalty": "4*normalized center distance + 0.25*abs(log area ratio)",
        },
        "limitation": "Lower confidence floors (0.005/0.001) and Top-20/50 are not recoverable from the saved proposal list, which was generated at conf=0.01 and capped at Top-10. This grid only reduces the saved proposal set; it does not test new low-confidence or additional proposals.",
        "interpretation": "Exploratory paired TRAIN diagnostic; no integration or model promotion decision.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(CONFIDENCE_FLOORS) * len(TOP_N_VALUES), "sequence_rows": len(sequence_rows_out), "summary_rows": len(summary_rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
