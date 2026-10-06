"""Score frozen label-blind tiled Viterbi predictions after the annotation freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001"
ORACLE = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001"
ANNOTATIONS = ORACLE / "frame_annotations_v1.csv"
OLD_RUNS = {
    "global_baseline": ORACLE / "full_baseline/frame_identity_results.csv",
    "global_oversampled": ORACLE / "full_oversampled/frame_identity_results.csv",
}
OLD_CENTER_RUNS = {
    "center_baseline": ORACLE / "center_baseline/frame_identity_results.csv",
    "center_oversampled": ORACLE / "center_oversampled/frame_identity_results.csv",
}
NEW_FRAME = DEFAULT_EXPERIMENT / "tiled_tracker_frame_predictions_v1.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_box(raw: str):
    if not raw:
        return None
    values = json.loads(raw)
    if len(values) != 4:
        raise ValueError(f"Invalid xyxy box {values}")
    return tuple(float(value) for value in values)


def iou(a, b) -> float:
    x1, y1, x2, y2 = a
    gx1, gy1, gx2, gy2 = b
    intersection = max(0.0, min(x2, gx2) - max(x1, gx1)) * max(0.0, min(y2, gy2) - max(y1, gy1))
    area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_b = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def score_frame(pred: dict[str, str], annotation: dict[str, str]) -> dict[str, object]:
    candidates = json.loads(pred.get("candidate_boxes_json", "[]") or "[]")
    visible = annotation["ball_visibility"] in {"VISIBLE", "PARTIAL"}
    target = parse_box(annotation.get("ball_of_play_box_json", "")) if visible else None
    distractors = [tuple(float(v) for v in box) for box in json.loads(annotation.get("distractor_boxes_json", "[]") or "[]")]
    candidate_boxes = [tuple(float(v) for v in item["box"]) for item in candidates]
    candidate_rank = next((index + 1 for index, box in enumerate(candidate_boxes) if target and iou(box, target) >= 0.5), 0)
    selected = parse_box(pred.get("selected_box_json", ""))
    target_iou = iou(selected, target) if selected and target else 0.0
    distractor_iou = max((iou(selected, box) for box in distractors), default=0.0) if selected else 0.0
    return {
        "target_annotated": target is not None,
        "target_candidate_rank": candidate_rank,
        "candidate_present": bool(candidate_rank),
        "identity_correct": target_iou >= 0.5,
        "selected_target_iou": f"{target_iou:.6f}",
        "wrong_ball_switch": distractor_iou >= 0.5,
        "selected_distractor_iou": f"{distractor_iou:.6f}",
        "candidate_count": len(candidate_boxes),
    }


def aggregate(rows: list[dict[str, object]], scope: str) -> dict[str, object]:
    targets = [row for row in rows if row["target_annotated"]]
    if not targets:
        raise ValueError(f"No target frames for {scope}")
    n = len(targets)
    return {
        "scope": scope, "target_frames": n,
        "candidate_hits": sum(bool(row["candidate_present"]) for row in targets),
        "candidate_recall": f"{sum(bool(row['candidate_present']) for row in targets) / n:.6f}",
        "identity_correct_frames": sum(bool(row["identity_correct"]) for row in targets),
        "identity_accuracy": f"{sum(bool(row['identity_correct']) for row in targets) / n:.6f}",
        "wrong_ball_switch_frames": sum(bool(row["wrong_ball_switch"]) for row in targets),
        "wrong_ball_switch_rate": f"{sum(bool(row['wrong_ball_switch']) for row in targets) / n:.6f}",
        "mean_proposals_per_target_frame": f"{sum(int(row['candidate_count']) for row in targets) / n:.6f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--tiled-frame-predictions", type=Path, default=NEW_FRAME)
    args = parser.parse_args()
    freeze_path = args.experiment_dir / "tiled_tracker_run_freeze_v1.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Prediction-only tracker outputs must be frozen before annotations are joined")
    prediction_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not prediction_freeze.get("prediction_only") or prediction_freeze.get("labels_included"):
        raise AssertionError("Prediction freeze is not label-blind")

    annotations = read_csv(args.annotations)
    if len(annotations) != 1435 or any(row["split"].lower() != "train" for row in annotations):
        raise ValueError("Expected exactly 1,435 TRAIN annotation rows")
    annotation_by_id = {row["sample_id"]: row for row in annotations}
    if len(annotation_by_id) != 1435:
        raise ValueError("Duplicate sample_id in frozen annotations")
    predictions_by_run: dict[str, list[dict[str, str]]] = {}
    for run, path in OLD_RUNS.items():
        predictions_by_run[run] = read_csv(path)
        if len(predictions_by_run[run]) != 1435:
            raise ValueError(f"Expected 1,435 frames in {path}")
    for run, path in OLD_CENTER_RUNS.items():
        predictions_by_run[run] = read_csv(path)
        if len(predictions_by_run[run]) != 385:
            raise ValueError(f"Expected 385 center-window frames in {path}")
    tiled_rows = read_csv(args.tiled_frame_predictions)
    if len(tiled_rows) != 2870:
        raise ValueError("Expected 2 x 1,435 frozen tiled tracker frame rows")
    for model in ("baseline", "oversampled"):
        subset = [row for row in tiled_rows if row["checkpoint"] == model]
        if len(subset) != 1435 or len({row["sample_id"] for row in subset}) != 1435:
            raise ValueError(f"Tiled tracker output incomplete for {model}")
        predictions_by_run[f"tiled_{model}"] = subset

    events = read_csv(ORACLE / "event_manifest_v1.csv")
    event_by_seq = {row["sequence_id"]: row for row in events}
    per_frame: list[dict[str, object]] = []
    per_sequence: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    scored: dict[tuple[str, str], dict[str, object]] = {}
    for run, predictions in predictions_by_run.items():
        pred_by_id = {row["sample_id"]: row for row in predictions}
        expected_rows = 385 if run.startswith("center_") else 1435
        if len(pred_by_id) != expected_rows:
            raise ValueError(f"Duplicate/missing sample rows in {run}: expected {expected_rows}")
        run_frame: list[dict[str, object]] = []
        by_seq: dict[str, list[dict[str, object]]] = defaultdict(list)
        for sample_id, annotation in annotation_by_id.items():
            if run.startswith("center_") and not 16 <= int(annotation["sequence_frame"]) <= 26:
                continue
            if sample_id not in pred_by_id:
                raise ValueError(f"Missing {sample_id} in {run}")
            result = score_frame(pred_by_id[sample_id], annotation)
            row = {
                "run": run, "sequence_id": annotation["sequence_id"],
                "sequence_frame": int(annotation["sequence_frame"]), "sample_id": sample_id,
                "event_id": annotation["event_id"], "event_role": annotation["event_role"],
                "pair_id": annotation["pair_id"], "ball_visibility": annotation["ball_visibility"],
                **result,
            }
            run_frame.append(row)
            by_seq[annotation["sequence_id"]].append(row)
            per_frame.append(row)
        scopes = {"center_16_26": run_frame} if run.startswith("center_") else {
            "full": run_frame,
            "center_16_26": [row for row in run_frame if 16 <= int(row["sequence_frame"]) <= 26],
        }
        for scope, scope_rows in scopes.items():
            summaries.append({"run": run, **aggregate(scope_rows, scope)})
            for role in ("CATALOGED_GOAL", "CONFIRMED_NO_GOAL"):
                role_rows = [row for row in scope_rows if row["event_role"] == role]
                if role_rows:
                    summaries.append({"run": run, "event_role": role, **aggregate(role_rows, scope)})
        for sequence_id, sequence_rows in by_seq.items():
            positive_rows = [row for row in sequence_rows if row["target_annotated"]]
            summary = aggregate(positive_rows, "sequence") if positive_rows else {"target_frames": 0}
            details = {
                "run": run, "sequence_id": sequence_id,
                "event_role": event_by_seq[sequence_id]["event_role"],
                "pair_id": event_by_seq[sequence_id]["pair_id"],
                **summary,
            }
            per_sequence.append(details)
            scored[(run, sequence_id)] = details

    pair_rows: list[dict[str, object]] = []
    by_pair: dict[str, dict[str, str]] = defaultdict(dict)
    for event in events:
        if event["pair_id"]:
            by_pair[event["pair_id"]][event["event_role"]] = event["sequence_id"]
    for run in predictions_by_run:
        for pair_id, pair in by_pair.items():
            if set(pair) != {"CATALOGED_GOAL", "CONFIRMED_NO_GOAL"}:
                continue
            positive = scored[(run, pair["CATALOGED_GOAL"])]
            control = scored[(run, pair["CONFIRMED_NO_GOAL"])]
            for metric in ("candidate_recall", "identity_accuracy"):
                pair_rows.append({
                    "run": run, "pair_id": pair_id, "metric": metric,
                    "positive_sequence": positive[metric], "control_sequence": control[metric],
                    "positive_minus_control": f"{float(positive[metric]) - float(control[metric]):.6f}",
                    "interpretation": "paired positive/control sequences; metrics use the frames available for this run",
                })

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.experiment_dir / "tiled_tracker_scored_summary_v1.csv", summaries)
    write_csv(args.experiment_dir / "tiled_tracker_scored_by_sequence_v1.csv", per_sequence)
    write_csv(args.experiment_dir / "tiled_tracker_scored_by_frame_v1.csv", per_frame)
    write_csv(args.experiment_dir / "tiled_tracker_scored_pairs_v1.csv", pair_rows)
    scored_manifest = {
        "experiment_id": args.experiment_dir.name,
        "prediction_freeze_sha256": sha256(freeze_path),
        "annotation_path": str(args.annotations.relative_to(ROOT)),
        "annotation_sha256": sha256(args.annotations),
        "old_full_window_predictions": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in OLD_RUNS.items()},
        "old_center_only_predictions": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in OLD_CENTER_RUNS.items()},
        "tiled_predictions": {"path": str(args.tiled_frame_predictions.relative_to(ROOT)), "sha256": sha256(args.tiled_frame_predictions)},
        "scoring": "Annotations were joined only after the label-blind tiled Viterbi outputs were frozen. Candidate recall uses any Top-10 proposal IoU>=0.50; identity uses selected-track IoU>=0.50; wrong switch uses selected IoU>=0.50 with any manual distractor. Center-only and full-window global runs are retained as separate references because their Viterbi contexts differ.",
        "scope": "TRAIN only; no VALIDATION/TEST; no V4 promotion",
    }
    (args.experiment_dir / "tiled_tracker_scoring_manifest_v1.json").write_text(json.dumps(scored_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(predictions_by_run), "summary_rows": len(summaries), "sequence_rows": len(per_sequence), "pair_rows": len(pair_rows), "annotation_sha256": scored_manifest["annotation_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
