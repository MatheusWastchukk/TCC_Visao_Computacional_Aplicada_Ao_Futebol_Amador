"""Measure BALL proposal-set geometry against frozen human GOAL regions/volumes."""
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
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tmp/ultralytics_ball_geometry_config"))
DEFAULT_ANNOTATIONS = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/frame_annotations_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/BALL-PROPOSAL-SET-GEOMETRY-EXP-0001"
CHECKPOINTS = {
    "baseline": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "oversampled": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
}
CONFIDENCES = (0.01, 0.005, 0.001)
OVERLAP = 0.20
DEDUP_IOU = 0.75


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


def parse_points(value: str) -> np.ndarray:
    points = np.asarray(json.loads(value), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"Invalid polygon points: {value}")
    return points


def make_tiles(frame) -> list[tuple[str, int, int, object]]:
    height, width = frame.shape[:2]
    tile_w = min(width, math.ceil(width / (2.0 - OVERLAP)))
    tile_h = min(height, math.ceil(height / (2.0 - OVERLAP)))
    return [
        (f"tile_{ri}{ci}", x, y, frame[y:y + tile_h, x:x + tile_w])
        for ri, y in enumerate((0, height - tile_h))
        for ci, x in enumerate((0, width - tile_w))
    ]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = a
    gx1, gy1, gx2, gy2 = b
    iw, ih = max(0.0, min(x2, gx2) - max(x1, gx1)), max(0.0, min(y2, gy2) - max(y1, gy1))
    inter = iw * ih
    union = max(0.0, x2 - x1) * max(0.0, y2 - y1) + max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1) - inter
    return inter / union if union > 0 else 0.0


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda r: -float(r["confidence"])):
        box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        if all(iou(box, tuple(float(other[key]) for key in ("x1", "y1", "x2", "y2"))) < DEDUP_IOU for other in kept):
            kept.append(row)
    return kept


def run(model: YOLO, image, imgsz: int, device: str, xoff: int = 0, yoff: int = 0) -> list[dict[str, object]]:
    result = model.predict(source=image, conf=min(CONFIDENCES), imgsz=imgsz, device=device, max_det=300, verbose=False)[0]
    if result.boxes is None:
        return []
    return [{
        "confidence": float(score), "x1": float(box[0]) + xoff, "y1": float(box[1]) + yoff,
        "x2": float(box[2]) + xoff, "y2": float(box[3]) + yoff,
    } for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True)]


def geometry_features(rows: list[dict[str, object]], opening: np.ndarray, volume: np.ndarray) -> dict[str, object]:
    opening_contour, volume_contour = opening.reshape((-1, 1, 2)), volume.reshape((-1, 1, 2))
    opening_diag = math.hypot(float(opening[:, 0].max() - opening[:, 0].min()), float(opening[:, 1].max() - opening[:, 1].min()))
    near_threshold = max(5.0, opening_diag * 0.02)
    centers = [((float(r["x1"]) + float(r["x2"])) / 2, (float(r["y1"]) + float(r["y2"])) / 2) for r in rows]
    opening_distances, volume_distances, in_opening, in_volume, near_opening = [], [], [], [], []
    for center in centers:
        point = (float(center[0]), float(center[1]))
        d_open = float(cv2.pointPolygonTest(opening_contour, point, True))
        d_volume = float(cv2.pointPolygonTest(volume_contour, point, True))
        opening_distances.append(max(0.0, -d_open))
        volume_distances.append(max(0.0, -d_volume))
        in_opening.append(d_open >= 0)
        in_volume.append(d_volume >= 0)
        near_opening.append(d_open >= -near_threshold)
    return {
        "proposal_count": len(rows),
        "min_distance_to_opening_px": f"{min(opening_distances, default=999999.0):.3f}",
        "min_distance_to_volume_px": f"{min(volume_distances, default=999999.0):.3f}",
        "max_conf_inside_opening": f"{max((float(r['confidence']) for r, inside in zip(rows, in_opening, strict=True) if inside), default=0.0):.8f}",
        "max_conf_inside_volume": f"{max((float(r['confidence']) for r, inside in zip(rows, in_volume, strict=True) if inside), default=0.0):.8f}",
        "n_proposals_inside_opening": sum(in_opening), "n_proposals_inside_volume": sum(in_volume),
        "n_proposals_near_opening": sum(near_opening), "opening_near_threshold_px": f"{near_threshold:.3f}",
    }


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

    all_rows = read_csv(args.annotations)
    rows = [row for row in all_rows if row["goal_region_status"] == "VALID" and row["goal_volume_status"] == "VALID"]
    if len(rows) != 35 or any(row["split"].lower() != "train" or row["is_event_frame"] != "True" for row in rows):
        raise ValueError("Expected the 35 completed TRAIN event frames with valid human GOAL region and volume")
    if sum(row["event_role"] == "CATALOGED_GOAL" for row in rows) != 29 or sum(row["event_role"] == "CONFIRMED_NO_GOAL" for row in rows) != 6:
        raise ValueError("Expected 29 cataloged goals and 6 confirmed NO_GOAL controls")
    for row in rows:
        image = Path(row["image_path"])
        row["_image_path"] = str(image if image.is_absolute() else ROOT / image)
        if not Path(row["_image_path"]).is_file():
            raise FileNotFoundError(row["_image_path"])

    detections: list[dict[str, object]] = []
    elapsed: dict[tuple[str, str], float] = defaultdict(float)
    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        model = YOLO(str(checkpoint_path))
        warm = cv2.imread(rows[0]["_image_path"])
        run(model, warm, args.imgsz, args.device)
        for index, annotation in enumerate(rows, 1):
            frame = cv2.imread(annotation["_image_path"])
            if frame is None:
                raise RuntimeError(f"Could not read {annotation['_image_path']}")
            views = [("global", 0, 0, frame), *(("tiles_2x2", x, y, crop) for _, x, y, crop in make_tiles(frame))]
            for view_name, xoff, yoff, view_image in views:
                started = time.perf_counter()
                found = run(model, view_image, args.imgsz, args.device, xoff, yoff)
                elapsed[(checkpoint_name, view_name)] += time.perf_counter() - started
                for item in found:
                    detections.append({
                        "sequence_id": annotation["sequence_id"], "event_id": annotation["event_id"],
                        "event_role": annotation["event_role"], "pair_id": annotation["pair_id"],
                        "matched_positive_id": annotation["matched_positive_id"], "match_id": annotation["match_id"],
                        "checkpoint": checkpoint_name, "view": view_name,
                        **item,
                    })
            if index % 10 == 0 or index == len(rows):
                print(f"{checkpoint_name}: {index}/{len(rows)} geometry frames", flush=True)
        del model

    by_seq: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in detections:
        by_seq[(str(item["checkpoint"]), str(item["sequence_id"]), str(item["view"]))].append(item)
    metric_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    frame_feature_rows: list[dict[str, object]] = []
    for checkpoint_name in CHECKPOINTS:
        for view in ("global", "tiles_2x2", "global_plus_tiles"):
            for confidence in CONFIDENCES:
                values_by_seq: dict[str, dict[str, object]] = {}
                for annotation in rows:
                    candidates = []
                    view_names = ("global",) if view == "global" else (("tiles_2x2",) if view == "tiles_2x2" else ("global", "tiles_2x2"))
                    for view_name in view_names:
                        candidates.extend(item for item in by_seq[(checkpoint_name, annotation["sequence_id"], view_name)] if float(item["confidence"]) >= confidence)
                    if view != "global":
                        candidates = deduplicate(candidates)
                    opening, volume = parse_points(annotation["goal_region_points_json"]), parse_points(annotation["goal_volume_points_json"])
                    features = geometry_features(candidates, opening, volume)
                    frame_feature_rows.append({
                        "checkpoint": checkpoint_name, "view": view, "confidence_threshold": confidence,
                        "sequence_id": annotation["sequence_id"], "event_id": annotation["event_id"],
                        "event_role": annotation["event_role"], "pair_id": annotation["pair_id"],
                        "matched_positive_id": annotation["matched_positive_id"], **features,
                    })
                    values_by_seq[annotation["sequence_id"]] = {**annotation, **features}

                paired: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
                for annotation in rows:
                    if annotation["pair_id"]:
                        paired[annotation["pair_id"]][annotation["event_role"]] = values_by_seq[annotation["sequence_id"]]
                features_names = (
                    "min_distance_to_opening_px", "min_distance_to_volume_px", "max_conf_inside_opening",
                    "max_conf_inside_volume", "n_proposals_inside_opening", "n_proposals_inside_volume",
                    "n_proposals_near_opening",
                )
                for feature in features_names:
                    wins = ties = losses = 0
                    for pair_id, pair in paired.items():
                        if set(pair) != {"CATALOGED_GOAL", "CONFIRMED_NO_GOAL"}:
                            continue
                        pos = float(pair["CATALOGED_GOAL"][feature])
                        neg = float(pair["CONFIRMED_NO_GOAL"][feature])
                        lower_better = feature.startswith("min_distance")
                        if math.isclose(pos, neg, abs_tol=1e-8):
                            outcome = "TIE"
                            ties += 1
                        elif (pos < neg) if lower_better else (pos > neg):
                            outcome = "POSITIVE_WINS"
                            wins += 1
                        else:
                            outcome = "CONTROL_WINS"
                            losses += 1
                        pair_rows.append({
                            "checkpoint": checkpoint_name, "view": view, "confidence_threshold": confidence,
                            "feature": feature, "pair_id": pair_id, "positive_value": pos,
                            "control_value": neg, "outcome": outcome,
                        })
                    metric_rows.append({
                        "checkpoint": checkpoint_name, "view": view, "confidence_threshold": confidence,
                        "feature": feature, "matched_pairs": wins + ties + losses,
                        "positive_wins": wins, "ties": ties, "control_wins": losses,
                        "unmatched_positive_frames": 23, "unpaired_controls": 0,
                    })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "proposal_set_geometry_by_frame_v1.csv", frame_feature_rows)
    write_csv(args.output / "proposal_set_geometry_pairwise_v1.csv", pair_rows)
    write_csv(args.output / "proposal_set_geometry_summary_v1.csv", metric_rows)
    write_csv(args.output / "proposal_set_geometry_raw_detections_v1.csv", detections)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN-only five-match holdout disjoint from both checkpoints; 35 geometry-marked event frames (29 goal, 6 paired controls); TEST/VALIDATION not accessed",
        "annotations": {"path": str(args.annotations.relative_to(ROOT)), "sha256": sha256(args.annotations)},
        "checkpoints": [{"name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in CHECKPOINTS.items()],
        "inference": {"imgsz": args.imgsz, "device": args.device, "confidence_floor": min(CONFIDENCES), "thresholds": list(CONFIDENCES), "max_det_per_view": 300,
                      "tile_grid": "2x2", "tile_overlap": OVERLAP, "cross_view_dedup_iou": DEDUP_IOU},
        "geometry": "Proposal-box center tested against manually marked opening quadrilateral and 6-point volume polygon using signed point-to-polygon distance. Near-opening threshold=max(5 px, 2% of opening bounding-box diagonal).",
        "limits": "Only the event frame has manual goal geometry; temporal aggregation, camera-motion compensation, and goal-structure tracking are not evaluated. Pairwise inference uses six matched positives/controls; 23 positive geometry frames are unmatched.",
        "runtime_seconds": {f"{model}:{view}": elapsed[(model, view)] / len(rows) for model in CHECKPOINTS for view in ("global", "tiles_2x2")},
        "interpretation": "Exploratory small-sample proposal-set geometry comparison; no external validation and no promotion decision.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"detections": len(detections), "frames": len(rows), "pairs": len({r['pair_id'] for r in rows if r['pair_id']}), "metrics": len(metric_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
