"""Summarize frozen two-detector Viterbi retests without tuning any setting."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
ANNOTATIONS = EXP / "frame_annotations_v1.csv"
EVENTS = EXP / "event_manifest_v1.csv"
FREEZE = EXP / "annotation_freeze_v1.json"
BLIND_RUN_FREEZE = EXP / "blind_tracker_run_freeze_v1.json"
RUNS = {
    ("center", "baseline"): EXP / "center_baseline",
    ("center", "oversampled"): EXP / "center_oversampled",
    ("full", "baseline"): EXP / "full_baseline",
    ("full", "oversampled"): EXP / "full_oversampled",
}
CHECKPOINTS = {
    "baseline": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "oversampled": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
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


def parse_box(raw: str) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    values = json.loads(raw)
    if len(values) != 4:
        raise ValueError(f"Expected xyxy box, got {values}")
    return tuple(float(value) for value in values)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def apply_frozen_labels(pred: dict[str, str], annotation: dict[str, str]) -> dict[str, str]:
    """Score label-blind frozen predictions only after annotation freeze."""
    if not pred.get("candidate_boxes_json"):
        raise ValueError(f"No frozen candidate list for {pred['sample_id']}")
    candidates = json.loads(pred["candidate_boxes_json"])
    visible = annotation["ball_visibility"] in {"VISIBLE", "PARTIAL"}
    target = parse_box(annotation.get("ball_of_play_box_json", "")) if visible else None
    distractors = [tuple(float(v) for v in box) for box in json.loads(annotation.get("distractor_boxes_json", "[]") or "[]")]
    candidate_boxes = [tuple(float(v) for v in item["box"]) for item in candidates]
    rank = next((i + 1 for i, box in enumerate(candidate_boxes) if target and iou(box, target) >= 0.5), 0)
    selected = parse_box(pred.get("selected_box_json", ""))
    target_iou = iou(selected, target) if selected and target else 0.0
    distractor_iou = max((iou(selected, box) for box in distractors), default=0.0) if selected else 0.0
    pred.update({
        "target_annotated": str(target is not None),
        "target_candidate_rank": str(rank),
        "selected_target_iou": f"{target_iou:.6f}",
        "selected_distractor_iou": f"{distractor_iou:.6f}",
        "identity_correct": str(target_iou >= 0.5),
        "wrong_ball_switch": str(distractor_iou >= 0.5),
    })
    return pred


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi_raw, yi_raw) in enumerate(polygon):
        xi, yi = float(xi_raw), float(yi_raw)
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def summarize_geometry(
    seq_frames: list[dict[str, str]],
    pred_frames: list[dict[str, str]],
    center_row: dict[str, str],
) -> dict[str, object]:
    if center_row.get("goal_region_status") != "VALID" or not center_row.get("goal_region_points_json"):
        return {
            "goal_region_status": center_row.get("goal_region_status", ""),
            "geometry_scored_frames": 0,
            "center_inside_aperture_frames": "",
            "center_inside_aperture_fraction": "",
            "aperture_ever_inside": "",
            "image_plane_entry_evidence": "",
            "image_plane_exit_evidence": "",
            "goal_volume_status": center_row.get("goal_volume_status", ""),
            "center_inside_volume_frames": "",
            "center_inside_volume_fraction": "",
        }
    aperture = json.loads(center_row["goal_region_points_json"])
    if len(aperture) != 4:
        raise ValueError(f"Expected four goal-region points at {center_row['event_id']}")
    volumes = json.loads(center_row["goal_volume_points_json"]) if center_row.get("goal_volume_status") == "VALID" and center_row.get("goal_volume_points_json") else None
    pred_by_sample = {row["sample_id"]: row for row in pred_frames}
    selected = []
    for row in seq_frames:
        if not 16 <= int(row["sequence_frame"]) <= 26:
            continue
        pred = pred_by_sample[row["sample_id"]]
        box = parse_box(pred.get("selected_box_json", ""))
        if box is None:
            selected.append((int(row["sequence_frame"]), None, None))
            continue
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        selected.append((int(row["sequence_frame"]), point_in_polygon(center, aperture), point_in_polygon(center, volumes) if volumes else None))
    aperture_values = [value for _, value, _ in selected if value is not None]
    volume_values = [value for _, _, value in selected if value is not None]
    inside_indices = [index for index, (_, inside, _) in enumerate(selected) if inside is True]
    first_inside, last_inside = (inside_indices[0], inside_indices[-1]) if inside_indices else (None, None)
    entry = bool(first_inside is not None and any(value is False for _, value, _ in selected[:first_inside]))
    exit_ = bool(last_inside is not None and any(value is False for _, value, _ in selected[last_inside + 1 :]))
    return {
        "goal_region_status": "VALID",
        "geometry_scored_frames": len(aperture_values),
        "center_inside_aperture_frames": sum(aperture_values),
        "center_inside_aperture_fraction": round(sum(aperture_values) / len(aperture_values), 6) if aperture_values else "",
        "aperture_ever_inside": bool(any(aperture_values)) if aperture_values else "",
        "image_plane_entry_evidence": entry if aperture_values else "",
        "image_plane_exit_evidence": exit_ if aperture_values else "",
        "goal_volume_status": center_row.get("goal_volume_status", ""),
        "center_inside_volume_frames": sum(volume_values) if volume_values else "",
        "center_inside_volume_fraction": round(sum(volume_values) / len(volume_values), 6) if volume_values else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--output-dir", type=Path, default=EXP)
    args = parser.parse_args()
    annotations = read_csv(ANNOTATIONS)
    events = read_csv(EVENTS)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if sha256(ANNOTATIONS) != freeze["annotations_sha256"]:
        raise ValueError("Annotation CSV changed after the requested annotation freeze.")
    if sha256(EVENTS) != freeze["event_manifest_sha256"]:
        raise ValueError("Event manifest changed after the requested annotation freeze.")
    blind_freeze = json.loads(BLIND_RUN_FREEZE.read_text(encoding="utf-8"))
    if not blind_freeze.get("prediction_only"):
        raise ValueError("Blind prediction freeze is missing or invalid.")
    for detector, checkpoint in CHECKPOINTS.items():
        if sha256(checkpoint) != blind_freeze["detector_checkpoints"][detector]["sha256"]:
            raise ValueError(f"Frozen detector checkpoint changed: {detector}")
    annotation_by_sample = {row["sample_id"]: row for row in annotations}
    event_by_sequence = {row["sequence_id"]: row for row in events}
    group_metrics: list[dict[str, object]] = []
    sequence_metrics: list[dict[str, object]] = []
    pair_values: dict[tuple[str, str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    run_hashes: dict[str, object] = {}

    for (window, detector), run_dir in RUNS.items():
        result_path = run_dir / "frame_identity_results.csv"
        summary_path = run_dir / "trajectory_identity_summary.csv"
        if not result_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing completed tracker run: {run_dir}")
        predictions = read_csv(result_path)
        run_metrics_path = run_dir / "metrics.json"
        run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
        if not run_metrics.get("prediction_only"):
            raise ValueError(f"Run is not marked label-blind prediction-only: {run_dir}")
        frozen_run = blind_freeze["runs"][run_dir.name]
        input_path = EXP / frozen_run["input_path"]
        if sha256(input_path) != frozen_run["input_sha256"] or sha256(result_path) != frozen_run["frame_results_sha256"]:
            raise ValueError(f"Blind inference input/output hash changed: {run_dir.name}")
        if sha256(summary_path) != frozen_run["trajectory_summary_sha256"] or sha256(run_metrics_path) != frozen_run["metrics_sha256"]:
            raise ValueError(f"Blind inference summary/metrics hash changed: {run_dir.name}")
        run_hashes[f"{window}_{detector}"] = {
            "frame_results_path": str(result_path.relative_to(ROOT)),
            "frame_results_sha256": sha256(result_path),
            "trajectory_summary_sha256": sha256(summary_path),
        }
        pred_by_seq: dict[str, list[dict[str, str]]] = defaultdict(list)
        aggregates: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for pred in predictions:
            ann = annotation_by_sample.get(pred["sample_id"])
            if ann is None:
                raise ValueError(f"Unknown sample_id in tracker output: {pred['sample_id']}")
            if pred["sequence_id"] != ann["sequence_id"]:
                raise ValueError(f"Sequence mismatch at {pred['sample_id']}")
            pred = apply_frozen_labels(pred, ann)
            pred_by_seq[pred["sequence_id"]].append(pred)
            if pred["target_annotated"] == "True":
                key = (ann["event_role"], ann["match_id"])
                aggregates[key]["target_frames"] += 1
                aggregates[key]["candidate_hits"] += pred["target_candidate_rank"] != "0"
                aggregates[key]["identity_hits"] += pred["identity_correct"] == "True"
                aggregates[key]["wrong_ball_switches"] += pred["wrong_ball_switch"] == "True"

        for (role, match), counts in sorted(aggregates.items()):
            n = counts["target_frames"]
            group_metrics.append({
                "window": window,
                "detector": detector,
                "event_role": role,
                "match_id": match,
                "target_frames": n,
                "candidate_hits": counts["candidate_hits"],
                "candidate_recall": round(counts["candidate_hits"] / n, 6) if n else "",
                "identity_hits": counts["identity_hits"],
                "identity_accuracy": round(counts["identity_hits"] / n, 6) if n else "",
                "wrong_ball_switches": counts["wrong_ball_switches"],
                "wrong_ball_switch_rate": round(counts["wrong_ball_switches"] / n, 6) if n else "",
            })

        for sequence_id, raw_sequence in sorted(pred_by_seq.items()):
            event = event_by_sequence[sequence_id]
            seq_preds = sorted(raw_sequence, key=lambda row: int(row["sequence_frame"]))
            seq_annotations = [annotation_by_sample[pred["sample_id"]] for pred in seq_preds]
            visible = [row["ball_visibility"] in {"VISIBLE", "PARTIAL"} for row in seq_annotations]
            candidate_miss = [is_visible and pred["target_candidate_rank"] == "0" for is_visible, pred in zip(visible, seq_preds)]
            identity_miss = [is_visible and pred["identity_correct"] != "True" for is_visible, pred in zip(visible, seq_preds)]
            selected = [pred["selected_candidate_rank"] != "0" for pred in seq_preds]
            central_center = next(row for row in seq_annotations if row["is_event_frame"] == "True")
            geom = summarize_geometry(seq_annotations, seq_preds, central_center)
            sequence_metric = {
                "window": window,
                "detector": detector,
                "sequence_id": sequence_id,
                "event_role": event["event_role"],
                "pair_id": event.get("pair_id", ""),
                "match_id": event["match_id"],
                "frames_scored": len(seq_preds),
                "visible_partial_target_frames": sum(visible),
                "candidate_miss_frames_on_visible_ball": sum(candidate_miss),
                "longest_candidate_miss_run_frames": longest_true_run(candidate_miss),
                "identity_error_frames_on_visible_ball": sum(identity_miss),
                "longest_identity_error_run_frames": longest_true_run(identity_miss),
                "selected_box_frames": sum(selected),
                "selected_box_fraction": round(sum(selected) / len(selected), 6) if selected else "",
                **geom,
            }
            sequence_metrics.append(sequence_metric)
            if event.get("pair_id"):
                pair_values[(window, detector, event["pair_id"])][event["event_role"]] = sequence_metric

    pair_rows: list[dict[str, object]] = []
    for (window, detector, pair_id), roles in sorted(pair_values.items()):
        positive = roles.get("CATALOGED_GOAL", {})
        negative = roles.get("CONFIRMED_NO_GOAL", {})
        if not positive or not negative:
            continue
        pos_frac = positive.get("center_inside_aperture_fraction", "")
        neg_frac = negative.get("center_inside_aperture_fraction", "")
        if pos_frac == "" or neg_frac == "":
            outcome = "NOT_EVALUABLE"
        elif math.isclose(float(pos_frac), float(neg_frac), rel_tol=0.0, abs_tol=1e-9):
            outcome = "TIE"
        else:
            outcome = "POSITIVE_HIGHER_APERTURE_OCCUPANCY" if float(pos_frac) > float(neg_frac) else "CONTROL_HIGHER_APERTURE_OCCUPANCY"
        pair_rows.append({
            "window": window,
            "detector": detector,
            "pair_id": pair_id,
            "match_id": positive["match_id"],
            "positive_sequence_id": positive["sequence_id"],
            "control_sequence_id": negative["sequence_id"],
            "positive_candidate_recall": "",
            "control_candidate_recall": "",
            "positive_identity_accuracy": "",
            "control_identity_accuracy": "",
            "positive_aperture_inside_fraction": pos_frac,
            "control_aperture_inside_fraction": neg_frac,
            "aperture_occupancy_outcome": outcome,
            "positive_image_plane_entry_evidence": positive.get("image_plane_entry_evidence", ""),
            "positive_image_plane_exit_evidence": positive.get("image_plane_exit_evidence", ""),
        })

    for row in pair_rows:
        for metric_row in sequence_metrics:
            if metric_row["window"] == row["window"] and metric_row["detector"] == row["detector"] and metric_row["sequence_id"] == row["positive_sequence_id"]:
                row["positive_candidate_recall"] = "" if not metric_row["visible_partial_target_frames"] else round(1 - metric_row["candidate_miss_frames_on_visible_ball"] / metric_row["visible_partial_target_frames"], 6)
                row["positive_identity_accuracy"] = "" if not metric_row["visible_partial_target_frames"] else round(1 - metric_row["identity_error_frames_on_visible_ball"] / metric_row["visible_partial_target_frames"], 6)
            elif metric_row["window"] == row["window"] and metric_row["detector"] == row["detector"] and metric_row["sequence_id"] == row["control_sequence_id"]:
                row["control_candidate_recall"] = "" if not metric_row["visible_partial_target_frames"] else round(1 - metric_row["candidate_miss_frames_on_visible_ball"] / metric_row["visible_partial_target_frames"], 6)
                row["control_identity_accuracy"] = "" if not metric_row["visible_partial_target_frames"] else round(1 - metric_row["identity_error_frames_on_visible_ball"] / metric_row["visible_partial_target_frames"], 6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tracker_retest_group_metrics_v1.csv", group_metrics)
    write_csv(args.output_dir / "tracker_retest_sequence_metrics_v1.csv", sequence_metrics)
    if pair_rows:
        write_csv(args.output_dir / "tracker_retest_pair_metrics_v1.csv", pair_rows)
    report = {
        "experiment_id": "ORACLE-TRACKER-RETEST-EXP-0001",
        "scope": "TRAIN-only engineering retest; no VALIDATION/TEST or V4 promotion",
        "annotation_freeze": {"path": str(args.freeze.resolve()), "sha256": sha256(args.freeze)},
        "annotation_scoring_scope": freeze.get("scoring_scope", "legacy/full"),
        "detector_checkpoints": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in CHECKPOINTS.items()},
        "tracker_configuration": {"confidence": 0.01, "max_candidates": 10, "imgsz": 960, "device": "cpu", "verifier": None},
        "runs": run_hashes,
        "scope_note": (
            "BALL-only freeze: center frames 16-26 are the candidate/identity score; full-window outputs support continuity and gaps. Goal geometry was not evaluated, and pairwise geometry outcomes are NOT_EVALUABLE until the missing goal-region/volume labels are completed."
            if freeze.get("ball_identity_only")
            else "Center window (frames 16-26) is the comparable candidate/identity score. Full-window outputs support continuity, gaps and descriptive image-plane geometry. Image-plane entry/exit flags are not physical 3D line-crossing claims."
        ),
        "outputs": [
            "tracker_retest_group_metrics_v1.csv",
            "tracker_retest_sequence_metrics_v1.csv",
            "tracker_retest_pair_metrics_v1.csv" if pair_rows else None,
        ],
    }
    report["annotation_freeze"] = {"path": str(args.freeze.resolve()), "sha256": sha256(args.freeze)}
    report["outputs"] = [
        "tracker_retest_group_metrics_v1.csv",
        "tracker_retest_sequence_metrics_v1.csv",
        "tracker_retest_pair_metrics_v1.csv" if pair_rows else None,
    ]
    report_path = args.output_dir / "tracker_retest_metrics_v1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(group_metrics), "sequences": len(sequence_metrics), "pairs": len(pair_rows), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
