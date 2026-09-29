"""Exploratory BALL input-size sweep on the manually labeled TRAIN audit frames."""
from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIOPEN_CACHE_ROOT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/miopen_workspace_cache_v1"
MIOPEN_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
for cache_subdir in ("user_db", "kernel_cache"):
    (MIOPEN_CACHE_ROOT / cache_subdir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(MIOPEN_CACHE_ROOT / "user_db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(MIOPEN_CACHE_ROOT / "kernel_cache"))

import cv2
import torch
from ultralytics import YOLO

RUN = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2"
OUT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/resolution_diagnostic_rocm_v2"
MODEL = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
SIZES = (960, 1280, 1536)
CONFIDENCE = 0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-12, area_a + area_b - inter)


def infer(device: str, samples: list[dict[str, str]], annotations: dict[str, dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    model = YOLO(str(MODEL))
    det_rows: list[dict[str, object]] = []
    timings = {str(size): 0.0 for size in SIZES}
    completed = 0
    for sample in samples:
        video_path = ROOT / "data/Videos_Edited" / sample["video_filename"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_index"]))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not decode {sample['sample_id']} from {video_path}")
        for size in SIZES:
            started = time.perf_counter()
            result = model.predict(source=frame, conf=CONFIDENCE, imgsz=size,
                                   device=(0 if device == "cuda:0" else "cpu"), verbose=False)[0]
            timings[str(size)] += time.perf_counter() - started
            boxes = []
            if result.boxes is not None:
                for coords, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True):
                    boxes.append((coords, float(conf)))
            boxes.sort(key=lambda item: -item[1])
            ann = annotations[sample["sample_id"]]
            gt = None
            if ann["ball_visibility"] == "BALL_VISIBLE":
                gt = tuple(float(ann[k]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2"))
            overlaps = [iou(gt, tuple(float(v) for v in coords)) for coords, _ in boxes] if gt else []
            hit30 = any(value >= 0.3 for value in overlaps)
            hit50 = any(value >= 0.5 for value in overlaps)
            matching_ranks = [rank for rank, value in enumerate(overlaps, 1) if value >= 0.5]
            false_count = len(boxes) if gt is None else max(0, len(boxes) - int(hit50))
            det_rows.append({
                "sample_id": sample["sample_id"], "match_id": sample["match_id"],
                "source_stratum": sample["source_stratum"], "imgsz": size,
                "ball_visibility": ann["ball_visibility"], "context_tags": ann.get("context_tags", ""),
                "proposal_count": len(boxes), "best_iou": f"{max(overlaps, default=0.0):.6f}" if gt else "",
                "hit_iou30": hit30 if gt else "", "hit_iou50": hit50 if gt else "",
                "true_rank_iou50": min(matching_ranks, default=""), "false_proposals_iou50": false_count,
                "proposal_details_json": json.dumps([
                    {"rank": rank, "confidence": conf,
                     "x1": float(coords[0]), "y1": float(coords[1]),
                     "x2": float(coords[2]), "y2": float(coords[3])}
                    for rank, (coords, conf) in enumerate(boxes, 1)
                ], separators=(",", ":")),
            })
        completed += 1
        if completed % 20 == 0 or completed == len(samples):
            print(f"{device}: {completed}/{len(samples)} frames", flush=True)
    total = sum(timings.values())
    return det_rows, {"inference_seconds_by_imgsz": timings,
                      "total_inference_seconds": total,
                      "mean_seconds_per_frame_all_sizes": total / max(1, len(samples))}


def summarize(samples: list[dict[str, str]], rows: list[dict[str, object]], runtime: dict[str, object], device: str) -> dict[str, object]:
    by_size = {}
    sequences: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in samples:
        if item.get("sequence_id"):
            sequences[item["sequence_id"]].append(item)
    for size in SIZES:
        group = [r for r in rows if int(r["imgsz"]) == size]
        visible = [r for r in group if r["ball_visibility"] == "BALL_VISIBLE"]
        evaluable = [r for r in group if r["ball_visibility"] in {"BALL_VISIBLE", "BALL_NOT_VISIBLE"}]
        entry: dict[str, object] = {
            "annotated_frames": len(group), "visible_frames": len(visible),
            "proposal_recall_iou30": sum(bool(r["hit_iou30"]) for r in visible) / max(1, len(visible)),
            "proposal_recall_iou50": sum(bool(r["hit_iou50"]) for r in visible) / max(1, len(visible)),
            "mean_false_proposals_per_evaluable_frame": sum(int(r["false_proposals_iou50"]) for r in evaluable) / max(1, len(evaluable)),
            "mean_proposals_per_frame": sum(int(r["proposal_count"]) for r in group) / max(1, len(group)),
            "rank1_share_among_iou50_matches": sum(str(r["true_rank_iou50"]) == "1" for r in visible) / max(1, sum(bool(r["hit_iou50"]) for r in visible)),
            "by_stratum": {},
        }
        for stratum in sorted({str(r["source_stratum"]) for r in group}):
            sg = [r for r in visible if r["source_stratum"] == stratum]
            entry["by_stratum"][stratum] = {
                "visible_frames": len(sg),
                "hit_iou30": sum(bool(r["hit_iou30"]) for r in sg),
                "hit_iou50": sum(bool(r["hit_iou50"]) for r in sg),
                "proposal_recall_iou50": sum(bool(r["hit_iou50"]) for r in sg) / max(1, len(sg)),
            }
        pair_total = pair_hit = 0
        rows_by_id = {str(r["sample_id"]): r for r in group}
        for sequence in sequences.values():
            ordered = sorted(sequence, key=lambda x: int(x["sequence_frame_index"]))
            for a, b in zip(ordered, ordered[1:]):
                aa, bb = rows_by_id[a["sample_id"]], rows_by_id[b["sample_id"]]
                delta = float(b["timestamp_seconds"]) - float(a["timestamp_seconds"])
                if aa["ball_visibility"] == bb["ball_visibility"] == "BALL_VISIBLE" and 0 < delta <= 0.25:
                    pair_total += 1
                    pair_hit += bool(aa["hit_iou50"]) and bool(bb["hit_iou50"])
        entry["adjacent_pairs_both_hit_iou50"] = pair_hit
        entry["adjacent_pairs_evaluable"] = pair_total
        entry["adjacent_pair_coverage_iou50"] = pair_hit / max(1, pair_total)
        by_size[str(size)] = entry
    return {"status": "completed", "scope": "TRAIN-only diagnostic; reuses the 400 manually annotated frames; TEST closed",
            "model": str(MODEL), "confidence": CONFIDENCE, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda:0" else None,
            "torch": torch.__version__, "miopen_user_db_path": __import__("os").environ.get("MIOPEN_USER_DB_PATH"),
            "miopen_custom_cache_dir": __import__("os").environ.get("MIOPEN_CUSTOM_CACHE_DIR"),
            "imgsz_values": list(SIZES), "runtime": runtime, "by_imgsz": by_size}


def main() -> int:
    samples = read_csv(RUN / "sample_manifest_v1.csv")
    annotations_list = read_csv(RUN / "manual_ball_review_v1.csv")
    annotations = {r["sample_id"]: r for r in annotations_list}
    if len(samples) != 400 or len(annotations) != 400 or any(s["split"] != "train" for s in samples):
        raise ValueError("Expected exactly 400 complete TRAIN frames; TEST remains closed.")
    if any(r["annotation_state"] != "complete" for r in annotations_list):
        raise ValueError("All manual annotations must be complete.")
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {OUT}")
    OUT.mkdir(parents=True)
    if torch.cuda.is_available():
        device = "cuda:0"
        try:
            rows, runtime = infer(device, samples, annotations)
        except Exception as exc:
            (OUT / "gpu_failure.txt").write_text(repr(exc) + "\n", encoding="utf-8")
            print(f"GPU inference failed; restarting the complete sweep on CPU: {exc!r}", flush=True)
            rows, runtime = infer("cpu", samples, annotations)
            device = "cpu"
            runtime["gpu_failure"] = repr(exc)
    else:
        device = "cpu"
        rows, runtime = infer(device, samples, annotations)
    write_csv(OUT / "per_frame_resolution_diagnostic_v1.csv", rows)
    result = summarize(samples, rows, runtime, device)
    (OUT / "resolution_diagnostic_metrics_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
