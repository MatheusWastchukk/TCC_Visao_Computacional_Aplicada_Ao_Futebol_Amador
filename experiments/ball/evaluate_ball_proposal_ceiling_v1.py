"""Measure confidence/Top-N proposal recall on the frozen TRAIN holdout fold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = ROOT / "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/holdout_train_annotations_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/BALL-PROPOSAL-CEILING-EXP-0001"
CHECKPOINTS = {
    "baseline": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "oversampled": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
}
CONFIDENCES = (0.01, 0.005, 0.001)
TOP_N = (10, 20, 50, None)
FIELDS = ("frame_id", "match_id", "checkpoint", "rank_by_confidence", "confidence", "x1", "y1", "x2", "y2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def iou(box: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    gx1, gy1, gx2, gy2 = gt
    iw, ih = max(0.0, min(x2, gx2) - max(x1, gx1)), max(0.0, min(y2, gy2) - max(y1, gy1))
    inter = iw * ih
    union = max(0.0, x2 - x1) * max(0.0, y2 - y1) + max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")
    for checkpoint in CHECKPOINTS.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    all_rows = read_rows(args.annotations)
    if len(all_rows) != 150 or {row["match_id"] for row in all_rows} != {"match_10", "match_16", "match_23", "match_27", "match_36"}:
        raise ValueError("Expected the exact frozen 150-frame TRAIN holdout across its five matches")
    if any(row["split"] != "holdout_train_match" or row["annotation_status"] != "completed" for row in all_rows):
        raise AssertionError("Holdout annotations must remain TRAIN-only and complete")
    frames = [row for row in all_rows if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}]
    if len(frames) != 104 or any(not row["ball_x1"] or not row["ball_y1"] or not row["ball_x2"] or not row["ball_y2"] for row in frames):
        raise ValueError("Expected 104 frozen VISIBLE/PARTIAL positives with GT boxes")
    for row in frames:
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = ROOT / image
        if not image.is_file():
            raise FileNotFoundError(image)
        row["_image_path"] = str(image)

    detections: list[dict[str, object]] = []
    progress: dict[str, dict[str, int]] = {}
    for checkpoint_name, checkpoint in CHECKPOINTS.items():
        model = YOLO(str(checkpoint))
        for index, row in enumerate(frames, 1):
            frame = cv2.imread(row["_image_path"])
            if frame is None:
                raise RuntimeError(f"Could not read image: {row['_image_path']}")
            result = model.predict(source=frame, conf=min(CONFIDENCES), imgsz=args.imgsz,
                                   device=args.device, max_det=300, verbose=False)[0]
            if result.boxes is None:
                continue
            proposals = sorted(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True), key=lambda pair: -float(pair[1]))
            frame_id = row["sample_id"]
            for rank, (box, score) in enumerate(proposals, 1):
                detections.append({
                    "frame_id": frame_id, "match_id": row["match_id"], "checkpoint": checkpoint_name,
                    "rank_by_confidence": rank, "confidence": f"{float(score):.8f}",
                    "x1": f"{float(box[0]):.3f}", "y1": f"{float(box[1]):.3f}",
                    "x2": f"{float(box[2]):.3f}", "y2": f"{float(box[3]):.3f}",
                })
            progress.setdefault(checkpoint_name, {})[frame_id] = len(proposals)
            if index % 20 == 0 or index == len(frames):
                print(f"{checkpoint_name}: {index}/{len(frames)} frames", flush=True)
        del model

    gt_by_frame = {
        row["sample_id"]: (float(row["ball_x1"]), float(row["ball_y1"]), float(row["ball_x2"]), float(row["ball_y2"]))
        for row in frames
    }
    by_frame: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for detection in detections:
        by_frame[(str(detection["checkpoint"]), str(detection["frame_id"]))].append(detection)

    metrics: list[dict[str, object]] = []
    per_frame: list[dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        for confidence in CONFIDENCES:
            for top_n in TOP_N:
                hits50 = hits30 = total_proposals = 0
                for row in frames:
                    found = [d for d in by_frame[(checkpoint, row["sample_id"])] if float(d["confidence"]) >= confidence]
                    found = found[:top_n] if top_n is not None else found
                    overlaps = [iou((float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])), gt_by_frame[row["sample_id"]]) for d in found]
                    hit50, hit30 = any(value >= 0.50 for value in overlaps), any(value >= 0.30 for value in overlaps)
                    hits50 += int(hit50)
                    hits30 += int(hit30)
                    total_proposals += len(found)
                    per_frame.append({
                        "checkpoint": checkpoint, "confidence_threshold": confidence,
                        "top_n": top_n if top_n is not None else "all",
                        "frame_id": row["sample_id"], "match_id": row["match_id"],
                        "proposals_kept": len(found), "best_iou": f"{max(overlaps, default=0.0):.6f}",
                        "hit_iou_050": hit50, "hit_iou_030": hit30,
                    })
                metrics.append({
                    "checkpoint": checkpoint, "confidence_threshold": confidence,
                    "top_n": top_n if top_n is not None else "all", "positive_frames": len(frames),
                    "candidate_recall_iou_050": f"{hits50 / len(frames):.6f}", "hits_iou_050": hits50,
                    "candidate_recall_iou_030": f"{hits30 / len(frames):.6f}", "hits_iou_030": hits30,
                    "mean_proposals_per_frame": f"{total_proposals / len(frames):.6f}",
                    "total_proposals": total_proposals,
                })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "proposal_ceiling_raw_detections_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(detections)
    with (args.output / "proposal_ceiling_metrics_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (args.output / "proposal_ceiling_by_frame_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    summary = {
        "experiment_id": args.output.name, "split": "TRAIN-only frozen holdout; five matches excluded from training; TEST not accessed",
        "annotation_path": str(args.annotations.relative_to(ROOT)), "annotation_sha256": sha256(args.annotations),
        "frames": 150, "positive_visible_or_partial_frames": len(frames), "matches": sorted({row['match_id'] for row in frames}),
        "checkpoints": [{"name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in CHECKPOINTS.items()],
        "inference": {"imgsz": args.imgsz, "device": args.device, "minimum_confidence": min(CONFIDENCES), "max_det": 300, "class_agnostic": False},
        "thresholds": list(CONFIDENCES), "top_n": [10, 20, 50, "all"],
        "warning": "The frozen TRAIN holdout supports an engineering threshold/candidate-ceiling analysis only; thresholds are exploratory and do not constitute external validation or a promotion decision.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"detections": len(detections), "metrics": len(metrics), "frames": len(frames)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
