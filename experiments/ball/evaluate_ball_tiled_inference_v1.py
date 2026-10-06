"""Compare global and overlapping tiled BALL inference on the frozen TRAIN holdout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tmp/ultralytics_ball_tiled_config"))
DEFAULT_ANNOTATIONS = ROOT / "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/holdout_train_annotations_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/BALL-TILED-INFERENCE-EXP-0001"
CHECKPOINTS = {
    "baseline": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "oversampled": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
}
CONFIDENCES = (0.01, 0.005, 0.001)
DEDUP_IOU = 0.75
OVERLAP = 0.20
VIEWS = ("global", "tiles_2x2", "global_plus_tiles")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = a
    gx1, gy1, gx2, gy2 = b
    iw, ih = max(0.0, min(x2, gx2) - max(x1, gx1)), max(0.0, min(y2, gy2) - max(y1, gy1))
    inter = iw * ih
    union = max(0.0, x2 - x1) * max(0.0, y2 - y1) + max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1) - inter
    return inter / union if union > 0 else 0.0


def tiles_2x2(frame, overlap: float) -> list[tuple[str, int, int, object]]:
    height, width = frame.shape[:2]
    tile_w = min(width, math.ceil(width / (2.0 - overlap)))
    tile_h = min(height, math.ceil(height / (2.0 - overlap)))
    xs = (0, width - tile_w)
    ys = (0, height - tile_h)
    result = []
    for row_i, y in enumerate(ys):
        for col_i, x in enumerate(xs):
            result.append((f"tile_{row_i}{col_i}", x, y, frame[y:y + tile_h, x:x + tile_w]))
    return result


def infer(model: YOLO, image, confidence: float, imgsz: int, device: str, offset_x: int = 0, offset_y: int = 0) -> list[dict[str, object]]:
    result = model.predict(source=image, conf=confidence, imgsz=imgsz, device=device, max_det=300, verbose=False)[0]
    if result.boxes is None:
        return []
    output = []
    for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True):
        output.append({
            "confidence": float(score), "x1": float(box[0]) + offset_x, "y1": float(box[1]) + offset_y,
            "x2": float(box[2]) + offset_x, "y2": float(box[3]) + offset_y,
        })
    return output


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: -float(item["confidence"])):
        box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        if all(box_iou(box, tuple(float(other[key]) for key in ("x1", "y1", "x2", "y2"))) < DEDUP_IOU for other in kept):
            kept.append(row)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    all_annotations = read_rows(args.annotations)
    if len(all_annotations) != 150 or {row["match_id"] for row in all_annotations} != {"match_10", "match_16", "match_23", "match_27", "match_36"}:
        raise ValueError("Expected the exact frozen 150-frame TRAIN holdout")
    if any(row["split"] != "holdout_train_match" or row["annotation_status"] != "completed" for row in all_annotations):
        raise AssertionError("Only completed TRAIN holdout annotations are accepted")
    annotations = [row for row in all_annotations if row["ball_visibility"] in {"VISIBLE", "PARTIAL", "NOT_VISIBLE"}]
    positive = {row["sample_id"] for row in annotations if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}}
    negative = {row["sample_id"] for row in annotations if row["ball_visibility"] == "NOT_VISIBLE"}
    if len(positive) != 104 or len(negative) != 44 or len(annotations) != 148:
        raise ValueError("Expected 104 positive, 44 NOT_VISIBLE negative and 2 excluded uncertain frames")
    for row in annotations:
        path = Path(row["image_path"])
        row["_image_path"] = str(path if path.is_absolute() else ROOT / path)
        if not Path(row["_image_path"]).is_file():
            raise FileNotFoundError(row["_image_path"])

    all_detections: list[dict[str, object]] = []
    elapsed: dict[tuple[str, str], float] = defaultdict(float)
    frame_counts: dict[tuple[str, str], int] = defaultdict(int)
    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        model = YOLO(str(checkpoint_path))
        warm_frame = cv2.imread(annotations[0]["_image_path"])
        infer(model, warm_frame, min(CONFIDENCES), args.imgsz, args.device)
        for frame_i, annotation in enumerate(annotations, 1):
            frame = cv2.imread(annotation["_image_path"])
            if frame is None:
                raise RuntimeError(f"Could not load {annotation['_image_path']}")
            views = [("global", 0, 0, frame), *(("tiles_2x2", x, y, crop) for _, x, y, crop in tiles_2x2(frame, OVERLAP))]
            detections_for_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
            for view_name, offset_x, offset_y, image in views:
                started = time.perf_counter()
                found = infer(model, image, min(CONFIDENCES), args.imgsz, args.device, offset_x, offset_y)
                elapsed[(checkpoint_name, view_name)] += time.perf_counter() - started
                frame_counts[(checkpoint_name, view_name)] += 1
                for detection in found:
                    detection.update({
                        "frame_id": annotation["sample_id"], "match_id": annotation["match_id"],
                        "checkpoint": checkpoint_name, "view": view_name,
                    })
                    detections_for_frame[view_name].append(detection)
                    all_detections.append(detection)
            if frame_i % 25 == 0 or frame_i == len(annotations):
                print(f"{checkpoint_name}: {frame_i}/{len(annotations)} frames", flush=True)
        del model

    gt = {
        row["sample_id"]: tuple(float(row[key]) for key in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"))
        for row in annotations if row["sample_id"] in positive
    }
    detections_by_frame_view: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for detection in all_detections:
        detections_by_frame_view[(str(detection["checkpoint"]), str(detection["frame_id"]), str(detection["view"]))].append(detection)

    metric_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for checkpoint_name in CHECKPOINTS:
        runtime_global = elapsed[(checkpoint_name, "global")] / len(annotations)
        runtime_tiles = elapsed[(checkpoint_name, "tiles_2x2")] / len(annotations)
        runtime = {"global": runtime_global, "tiles_2x2": runtime_tiles, "global_plus_tiles": runtime_global + runtime_tiles}
        for view in VIEWS:
            for confidence in CONFIDENCES:
                positives_050 = positives_030 = frames_with_proposal = negative_frames_with_proposal = proposal_total = 0
                for annotation in annotations:
                    frame_id = annotation["sample_id"]
                    found: list[dict[str, object]] = []
                    view_names = ("global",) if view == "global" else (("tiles_2x2",) if view == "tiles_2x2" else ("global", "tiles_2x2"))
                    for view_name in view_names:
                        found.extend(row for row in detections_by_frame_view[(checkpoint_name, frame_id, view_name)] if float(row["confidence"]) >= confidence)
                    if view != "global":
                        found = deduplicate(found)
                    proposal_total += len(found)
                    has_any = bool(found)
                    frames_with_proposal += int(has_any)
                    if frame_id in negative:
                        negative_frames_with_proposal += int(has_any)
                    if frame_id in positive:
                        overlaps = [box_iou(tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")), gt[frame_id]) for row in found]
                        hit050 = any(value >= 0.50 for value in overlaps)
                        hit030 = any(value >= 0.30 for value in overlaps)
                        positives_050 += int(hit050)
                        positives_030 += int(hit030)
                        frame_rows.append({
                            "checkpoint": checkpoint_name, "view": view, "confidence_threshold": confidence,
                            "frame_id": frame_id, "match_id": annotation["match_id"],
                            "proposals": len(found), "best_iou": f"{max(overlaps, default=0.0):.6f}",
                            "hit_iou_050": hit050, "hit_iou_030": hit030,
                        })
                metric_rows.append({
                    "checkpoint": checkpoint_name, "view": view, "confidence_threshold": confidence,
                    "positive_frames": len(positive), "candidate_recall_iou_050": f"{positives_050 / len(positive):.6f}",
                    "hits_iou_050": positives_050, "candidate_recall_iou_030": f"{positives_030 / len(positive):.6f}",
                    "hits_iou_030": positives_030, "all_labeled_frames": len(annotations),
                    "frames_with_any_proposal": frames_with_proposal,
                    "not_visible_negative_frames_with_proposal": negative_frames_with_proposal,
                    "mean_proposals_per_labeled_frame": f"{proposal_total / len(annotations):.6f}",
                    "mean_proposals_per_positive_frame": f"{proposal_total / len(positive):.6f}",
                    "mean_inference_seconds_per_frame": f"{runtime[view]:.6f}",
                })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "tiled_inference_raw_detections_v1.csv", all_detections)
    write_csv(args.output / "tiled_inference_metrics_v1.csv", metric_rows)
    write_csv(args.output / "tiled_inference_positive_by_frame_v1.csv", frame_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN-only frozen holdout; 5 matches; 104 positive and 44 NOT_VISIBLE frames; two UNCERTAIN frames excluded; TEST/VALIDATION not accessed",
        "annotations": {"path": str(args.annotations.relative_to(ROOT)), "sha256": sha256(args.annotations)},
        "checkpoints": [{"name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in CHECKPOINTS.items()],
        "imgsz": args.imgsz, "device": args.device, "inference_confidence_floor": min(CONFIDENCES),
        "thresholds_evaluated": list(CONFIDENCES), "max_det_per_view": 300,
        "tiling": {"grid": "2x2", "nominal_overlap_fraction": OVERLAP, "tile_size_formula": "ceil(image_dimension/(2-overlap)); starts at each border"},
        "cross_view_dedup_iou": DEDUP_IOU,
        "mean_inference_seconds_per_frame": {
            f"{model}:global": elapsed[(model, "global")] / len(annotations)
            for model in CHECKPOINTS
        } | {
            f"{model}:tiles_2x2": elapsed[(model, "tiles_2x2")] / len(annotations)
            for model in CHECKPOINTS
        },
        "timing_scope": "Model predict wall time only; image decoding/crop assembly/output I/O excluded; one warmup prediction per checkpoint excluded.",
        "warning": "Fixed TRAIN holdout engineering experiment; not external validation or a promotion decision. Cross-view confidence pooling is not used to rank proposals; all surviving proposals are evaluated.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"detections": len(all_detections), "metrics": len(metric_rows), "positive_frames": len(positive), "negative_frames": len(negative)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
