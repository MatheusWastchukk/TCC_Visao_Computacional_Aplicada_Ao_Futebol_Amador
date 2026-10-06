"""Compare the two frozen detector checkpoints on the untouched TRAIN holdout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MATCHES = {"match_10", "match_16", "match_23", "match_27", "match_36"}
AREA_ADDENDUM = Path(__file__).with_name("protocol_addendum_area_stratification_v1.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oversampled", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path(__file__).parent / "holdout_train_annotations_v1.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "holdout_evaluation_v1")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    args.output = args.output.resolve()
    annotation_path = args.annotations.resolve()
    annotation_sha256 = sha256(annotation_path)
    rows = read_csv(annotation_path)
    if len(rows) != 150 or {r["match_id"] for r in rows} != EXPECTED_MATCHES:
        raise ValueError("Holdout mismatch: expected exactly 150 rows from the five frozen TRAIN matches.")
    if any(r["split"] != "holdout_train_match" for r in rows):
        raise ValueError("Only the frozen holdout_train_match annotation rows are allowed.")
    if any("test" in r["split"].casefold() or "validation" in r["split"].casefold() for r in rows):
        raise AssertionError("VALIDATION/TEST rows are forbidden.")
    positive_rows = [r for r in rows if r["ball_visibility"] in {"VISIBLE", "PARTIAL"}]
    positive_areas = [
        max(0.0, float(row["ball_x2"]) - float(row["ball_x1"]))
        * max(0.0, float(row["ball_y2"]) - float(row["ball_y1"]))
        for row in positive_rows
    ]
    area_group_counts = {
        "lt_400_px2": sum(area < 400.0 for area in positive_areas),
        "ge_400_px2": sum(area >= 400.0 for area in positive_areas),
    }
    if len(positive_rows) != 104 or area_group_counts != {"lt_400_px2": 9, "ge_400_px2": 95}:
        raise ValueError(f"Frozen holdout positive-area groups differ from the addendum: {area_group_counts}")

    checkpoint_paths = {"baseline": args.baseline.resolve(), "oversampled": args.oversampled.resolve()}
    results: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for model_id, checkpoint in checkpoint_paths.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint))
        first = cv2.imread(str(ROOT / rows[0]["image_path"]))
        if first is None:
            raise FileNotFoundError(rows[0]["image_path"])
        warm_start = time.perf_counter()
        model.predict(source=first, conf=args.confidence, imgsz=args.imgsz, max_det=args.top_n, device="cpu", verbose=False)
        warmup_seconds = time.perf_counter() - warm_start

        model_rows: list[dict[str, object]] = []
        durations: list[float] = []
        for idx, row in enumerate(rows, start=1):
            image_path = ROOT / row["image_path"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            started = time.perf_counter()
            prediction = model.predict(
                source=image,
                conf=args.confidence,
                imgsz=args.imgsz,
                max_det=args.top_n,
                device="cpu",
                verbose=False,
            )[0]
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            boxes = prediction.boxes.xyxy.tolist() if prediction.boxes is not None else []
            confs = prediction.boxes.conf.tolist() if prediction.boxes is not None else []
            order = sorted(range(len(boxes)), key=lambda i: (-float(confs[i]), i))[: args.top_n]
            boxes = [tuple(float(v) for v in boxes[i]) for i in order]
            if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
                gt = tuple(float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"))
                gt_area_px2 = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
                overlaps = [iou(gt, box) for box in boxes]
            else:
                gt_area_px2 = None
                overlaps = []
            matching = [i + 1 for i, overlap in enumerate(overlaps) if overlap >= args.iou]
            out = {
                "model_id": model_id,
                "sample_id": row["sample_id"],
                "match_id": row["match_id"],
                "ball_visibility": row["ball_visibility"],
                "gt_area_px2": f"{gt_area_px2:.3f}" if gt_area_px2 is not None else "",
                "candidate_count_top_n": len(boxes),
                "best_iou": f"{max(overlaps, default=0.0):.6f}",
                "matching_rank": matching[0] if matching else 0,
                "candidate_hit": bool(matching),
                "inference_seconds": f"{elapsed:.6f}",
            }
            model_rows.append(out)
            results.append(out)
            if idx % 25 == 0 or idx == len(rows):
                print(f"{model_id}: {idx}/{len(rows)}")

        positive = [r for r in model_rows if r["ball_visibility"] in {"VISIBLE", "PARTIAL"}]
        nonvisible = [r for r in model_rows if r["ball_visibility"] == "NOT_VISIBLE"]
        groupings: list[tuple[str, str, list[dict[str, object]]]] = [("all_visible_partial", "all", positive)]
        for visibility in ("VISIBLE", "PARTIAL"):
            groupings.append(("visibility", visibility, [r for r in positive if r["ball_visibility"] == visibility]))
        for match in sorted(EXPECTED_MATCHES):
            groupings.append(("match", match, [r for r in positive if r["match_id"] == match]))
        groupings.extend([
            ("area_bucket", "lt_400_px2", [r for r in positive if float(r["gt_area_px2"]) < 400.0]),
            ("area_bucket", "ge_400_px2", [r for r in positive if float(r["gt_area_px2"]) >= 400.0]),
        ])
        for group_type, group_value, group_rows in groupings:
            hits = sum(bool(r["candidate_hit"]) for r in group_rows)
            summaries.append({
                "model_id": model_id,
                "group_type": group_type,
                "group_value": group_value,
                "positive_frames": len(group_rows),
                "hits_top_n_iou50": hits,
                "candidate_recall": f"{hits / len(group_rows):.6f}" if group_rows else "",
                "mean_proposals_top_n": f"{statistics.mean(int(r['candidate_count_top_n']) for r in group_rows):.3f}" if group_rows else "",
            })
        summaries.append({
            "model_id": model_id,
            "group_type": "NOT_VISIBLE_diagnostic",
            "group_value": "all",
            "positive_frames": len(nonvisible),
            "hits_top_n_iou50": "",
            "candidate_recall": "",
            "mean_proposals_top_n": f"{statistics.mean(int(r['candidate_count_top_n']) for r in nonvisible):.3f}" if nonvisible else "",
        })
        summaries.append({
            "model_id": model_id,
            "group_type": "runtime",
            "group_value": "CPU_after_one_warmup",
            "positive_frames": len(model_rows),
            "hits_top_n_iou50": "",
            "candidate_recall": "",
            "mean_proposals_top_n": f"{statistics.mean(int(r['candidate_count_top_n']) for r in model_rows):.3f}",
            "warmup_seconds": f"{warmup_seconds:.6f}",
            "mean_inference_ms": f"{1000 * statistics.mean(durations):.3f}",
            "p50_inference_ms": f"{1000 * statistics.median(durations):.3f}",
            "p95_inference_ms": f"{1000 * sorted(durations)[int(0.95 * (len(durations)-1))]:.3f}",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        })

    args.output.mkdir(parents=True, exist_ok=False)
    for filename, data in (("holdout_frame_results_v1.csv", results), ("holdout_summary_v1.csv", summaries)):
        with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in data for key in row)))
            writer.writeheader()
            writer.writerows(data)
    protocol = {
        "experiment_id": "BALL-HARD-POSITIVE-EXP-0001",
        "scope": "five untouched TRAIN matches only; not VALIDATION or TEST and not an external generalization estimate",
        "holdout_matches": sorted(EXPECTED_MATCHES),
        "holdout_rows": len(rows),
        "annotation_source": {
            "path": str(annotation_path),
            "sha256": annotation_sha256,
        },
        "comparison": "baseline versus one 3x small-positive oversampling intervention; same saved last.pt rule",
        "confidence": args.confidence,
        "top_n": args.top_n,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "device": "CPU",
        "candidate_recall": "fraction of VISIBLE/PARTIAL frames with any of top-N predictions at IoU >= threshold",
        "positive_area_threshold_px2": 400.0,
        "positive_area_group_counts": area_group_counts,
        "area_stratification": "Descriptive diagnostic only: candidate recall for positive GT boxes <400 px2 and >=400 px2; does not change the overall tracker gate or select checkpoints.",
        "analysis_addendum": {
            "path": str(AREA_ADDENDUM.relative_to(ROOT)),
            "sha256": sha256(AREA_ADDENDUM),
        },
        "evaluation_artifacts": {
            filename: {
                "path": str((args.output / filename).resolve()),
                "sha256": sha256(args.output / filename),
            }
            for filename in ("holdout_frame_results_v1.csv", "holdout_summary_v1.csv")
        },
        "negative_diagnostic": "proposal count on NOT_VISIBLE frames; no false-negative or recall label is assigned to UNCERTAIN rows",
        "selection": "No checkpoint or configuration selection is permitted from these holdout outcomes.",
        "checkpoints": {name: {"path": str(path), "sha256": sha256(path)} for name, path in checkpoint_paths.items()},
    }
    (args.output / "protocol_v1.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
