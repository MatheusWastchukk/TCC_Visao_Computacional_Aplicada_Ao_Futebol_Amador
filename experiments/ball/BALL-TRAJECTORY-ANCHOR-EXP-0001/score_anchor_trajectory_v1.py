"""Score frozen Viterbi and anchored trajectory outputs against manual TRAIN labels."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
TRACK = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001"
ANNOTATIONS = TRACK / "frame_annotations_v1.csv"
BLIND_FREEZE = EXP / "blind_replay_freeze_v1.json"
THRESHOLD = 0.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not BLIND_FREEZE.exists():
        raise FileNotFoundError("Run the label-blind replay before scoring.")
    freeze = json.loads(BLIND_FREEZE.read_text(encoding="utf-8"))
    protocol_path = EXP / "protocol_v1.json"
    source_freeze_path = TRACK / "blind_tracker_run_freeze_v1.json"
    if sha256(protocol_path) != freeze["protocol_sha256"]:
        raise ValueError("Experiment protocol changed after the blind replay freeze.")
    if sha256(source_freeze_path) != freeze["source_prediction_freeze_sha256"]:
        raise ValueError("Source prediction freeze changed after the blind replay.")
    for run_meta in freeze["runs"].values():
        anchor_path = ROOT / run_meta["anchor_predictions_path"]
        if sha256(anchor_path) != run_meta["anchor_predictions_sha256"]:
            raise ValueError(f"Frozen anchored predictions changed: {anchor_path}")
    annotations = read_csv(ANNOTATIONS)
    pending = [r["sample_id"] for r in annotations if r["annotation_status"] != "completed"]
    if pending:
        raise ValueError(f"Manual BALL review is incomplete: {len(pending)} frames remain; no labels read for scoring.")
    labels = {r["sample_id"]: r for r in annotations}
    input_rows = read_csv(TRACK / "blind_tracker_inputs_full_v1.csv")
    dimensions: dict[str, float] = {}
    import cv2
    for row in input_rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        dimensions[row["sample_id"]] = math.hypot(image.shape[0], image.shape[1])

    per_frame: list[dict[str, object]] = []
    for run_name, run_meta in freeze["runs"].items():
        window, detector = run_name.rsplit("_", 1)
        baseline_path = TRACK / f"{window}_{detector}" / "frame_identity_results.csv"
        anchor_path = ROOT / run_meta["anchor_predictions_path"]
        baseline_rows = {r["sample_id"]: r for r in read_csv(baseline_path)}
        anchor_rows = {r["sample_id"]: r for r in read_csv(anchor_path)}
        selected_by_sequence: dict[str, list[tuple[int, list[float] | None, str]]] = defaultdict(list)
        for row in input_rows if window == "full" else read_csv(TRACK / "blind_tracker_inputs_center_v1.csv"):
            sample_id = row["sample_id"]
            base = baseline_rows[sample_id]
            anchor = anchor_rows[sample_id]
            label = labels[sample_id]
            visibility = label["ball_visibility"].upper()
            target = json.loads(label["ball_of_play_box_json"]) if label["ball_of_play_box_json"] else None
            distractors = json.loads(label["distractor_boxes_json"] or "[]")
            candidates = json.loads(base["candidate_boxes_json"] or "[]")
            viterbi = json.loads(base["selected_box_json"]) if base["selected_box_json"] else None
            anchored = json.loads(anchor["selected_box_json"]) if anchor["selected_box_json"] else None
            visible = visibility in {"VISIBLE", "PARTIAL"} and target is not None
            candidate_recall = any(iou(c["box"], target) >= THRESHOLD for c in candidates) if visible else None
            base_iou = iou(viterbi, target) if visible and viterbi else (0.0 if visible else None)
            anchor_iou = iou(anchored, target) if visible and anchored else (0.0 if visible else None)
            base_distractor = max((iou(viterbi, d) for d in distractors), default=0.0) if viterbi else 0.0
            anchor_distractor = max((iou(anchored, d) for d in distractors), default=0.0) if anchored else 0.0
            frame_number = int(row["sequence_frame"])
            primary = window == "full" and 16 <= frame_number <= 26
            for method, box, target_iou, distractor_iou in (
                ("VITERBI", viterbi, base_iou, base_distractor),
                ("ANCHOR_TRAJECTORY", anchored, anchor_iou, anchor_distractor),
            ):
                per_frame.append({
                    "run": run_name, "window": window, "detector": detector, "method": method,
                    "sequence_id": row["sequence_id"], "sequence_frame": frame_number,
                    "sample_id": sample_id, "event_role": row["event_role"], "match_id": row["match_id"],
                    "primary_center_frame": primary, "visibility": visibility,
                    "candidate_recalled_iou50": candidate_recall,
                    "selected_box_present": box is not None,
                    "identity_correct_iou50": target_iou >= THRESHOLD if target_iou is not None else None,
                    "selected_target_iou": target_iou,
                    "distractor_switch_iou50": bool(box is not None and distractor_iou >= THRESHOLD),
                    "selected_box_json": json.dumps(box) if box else "",
                    "anchor_status": anchor["anchor_status"] if method == "ANCHOR_TRAJECTORY" else "",
                })
        # Adjacent-frame displacement normalized by image diagonal and frame gap.
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for record in per_frame:
            if record["run"] == run_name:
                grouped[(str(record["method"]), str(record["sequence_id"]))].append(record)
        for (method, _sequence), seq in grouped.items():
            seq.sort(key=lambda r: int(r["sequence_frame"]))
            previous: dict[str, object] | None = None
            for record in seq:
                jump = None
                if previous and previous["selected_box_json"] and record["selected_box_json"]:
                    a = json.loads(str(previous["selected_box_json"]))
                    b = json.loads(str(record["selected_box_json"]))
                    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
                    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    delta = int(record["sequence_frame"]) - int(previous["sequence_frame"])
                    jump = math.hypot(bx - ax, by - ay) / dimensions[str(record["sample_id"])] / delta
                record["normalized_jump_per_frame"] = jump
                previous = record

    frame_path = EXP / "scored_frame_results_v1.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    metrics: list[dict[str, object]] = []
    keys = sorted({(str(r["run"]), str(r["method"])) for r in per_frame})
    for run_name, method in keys:
        rows = [r for r in per_frame if r["run"] == run_name and r["method"] == method]
        window = str(rows[0]["window"])
        for scope, subset in (("FULL_WINDOW", rows), ("PRIMARY_CENTER_16_26", [r for r in rows if r["primary_center_frame"]])):
            if scope.startswith("PRIMARY") and window != "full":
                continue
            visible = [r for r in subset if r["identity_correct_iou50"] is not None]
            candidates = [r for r in subset if r["candidate_recalled_iou50"] is not None]
            jumps = [float(r["normalized_jump_per_frame"]) for r in subset if r.get("normalized_jump_per_frame") is not None]
            metrics.append({
                "run": run_name, "scope": scope, "method": method,
                "frames": len(subset), "visible_or_partial_frames": len(visible),
                "candidate_recall_iou50": mean([float(r["candidate_recalled_iou50"]) for r in candidates]) if candidates else None,
                "identity_accuracy_iou50": mean([float(r["identity_correct_iou50"]) for r in visible]) if visible else None,
                "selected_box_coverage_on_visible": mean([float(r["selected_box_present"]) for r in visible]) if visible else None,
                "distractor_switch_rate_iou50_on_visible": mean([float(r["distractor_switch_iou50"]) for r in visible]) if visible else None,
                "mean_normalized_jump_per_frame": mean(jumps),
                "frames_with_selected_box": sum(bool(r["selected_box_present"]) for r in subset),
                "track_lost_frames": sum(r.get("anchor_status") == "TRACK_LOST_MAX_GAP" for r in subset) if method == "ANCHOR_TRAJECTORY" else None,
            })
    summary_path = EXP / "scored_summary_v1.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps({"status": "SCORED_TRAIN_ONLY", "frames_csv": str(frame_path), "summary_csv": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
