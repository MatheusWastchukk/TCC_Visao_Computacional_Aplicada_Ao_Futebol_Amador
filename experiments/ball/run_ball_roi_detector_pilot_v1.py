"""Exploratory ROI BALL detector pilot on the existing 400 TRAIN audit frames."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001"
CACHE = EXP / "miopen_workspace_cache_v1"
(CACHE / "user_db").mkdir(parents=True, exist_ok=True)
(CACHE / "kernel_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(CACHE / "user_db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(CACHE / "kernel_cache"))

import cv2
import torch
from ultralytics import YOLO

RUN = EXP / "run_v2"
OUT = EXP / "roi_ball_detector_pilot_v2"
MODEL_PATH = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
CONF = 0.05
IOU_DEDUP = 0.5
ROI_MARGIN = (2.0, 3.0, 2.0)  # lateral, above, below GOAL box dimensions


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(1e-12, area_a + area_b - inter)


def infer(model: YOLO, image, size: int, device: str) -> tuple[list[dict], float]:
    if device == "cuda:0":
        torch.cuda.synchronize()
    started = time.perf_counter()
    pred = model.predict(source=image, conf=CONF, imgsz=size,
                         device=(0 if device == "cuda:0" else "cpu"), verbose=False)[0]
    if device == "cuda:0":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    detections = []
    if pred.boxes is not None:
        for box, confidence in zip(pred.boxes.xyxy.tolist(), pred.boxes.conf.tolist(), strict=True):
            detections.append({"box": tuple(float(v) for v in box), "confidence": float(confidence)})
    detections.sort(key=lambda d: -d["confidence"])
    return detections, elapsed


def deduplicate(detections: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for item in sorted(detections, key=lambda d: -d["confidence"]):
        if not any(iou(item["box"], prior["box"]) >= IOU_DEDUP for prior in kept):
            kept.append(item)
    return kept


def roi_from_goal(goal: dict | None, width: int, height: int):
    if goal is None:
        return None
    gx1, gy1, gx2, gy2 = (float(goal[k]) for k in ("x1", "y1", "x2", "y2"))
    gw, gh = max(1.0, gx2-gx1), max(1.0, gy2-gy1)
    lat, top, bottom = ROI_MARGIN
    return (max(0, int(gx1-lat*gw)), max(0, int(gy1-top*gh)),
            min(width, int(gx2+lat*gw)), min(height, int(gy2+bottom*gh)))


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    manifest = read_csv(RUN / "sample_manifest_v1.csv")
    annotations = {r["sample_id"]: r for r in read_csv(RUN / "manual_ball_review_v1.csv")}
    raw = read_csv(RUN / "raw_detections_v1.csv")
    if len(manifest) != 400 or len(annotations) != 400 or any(r["split"] != "train" for r in manifest):
        raise ValueError("Expected exactly the existing 400 annotated TRAIN frames.")
    goals: dict[str, list[dict[str, str]]] = {}
    for row in raw:
        if row["detector"] == "GOAL":
            goals.setdefault(row["sample_id"], []).append(row)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda:0" else None
    metrics_rows: list[dict] = []
    failure = None
    try:
        metrics_rows, timings = execute(manifest, annotations, goals, device)
    except Exception as exc:
        if device == "cpu":
            raise
        failure = f"{type(exc).__name__}: {exc}"
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        device = "cpu"
        metrics_rows, timings = execute(manifest, annotations, goals, device)

    OUT.mkdir(parents=True)
    with (OUT / "per_frame_metrics_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = summarize(metrics_rows)
    result = {
        "status": "completed",
        "scope": "exploratory TRAIN pilot; reuses 400 frames selected and annotated before this pilot; TEST closed",
        "model": str(MODEL_PATH), "confidence": CONF, "device": device,
        "gpu": gpu_name if device == "cuda:0" else None,
        "torch": torch.__version__, "hip": torch.version.hip,
        "roi": {"source": "top-confidence GOAL box from frozen run_v2", "confidence": 0.25,
                "imgsz": 640, "lateral_goal_widths": ROI_MARGIN[0],
                "above_goal_heights": ROI_MARGIN[1], "below_goal_heights": ROI_MARGIN[2]},
        "runtime_seconds": timings,
        "gpu_failure_before_cpu_fallback": failure,
        "deduplication": f"hybrid proposal union uses confidence-sorted cross-source IoU suppression at {IOU_DEDUP}",
        "by_mode": summary,
        "limitations": [
            "The 400 frames were reused after selecting 1536 and ROI dimensions; this is not independent confirmation.",
            "ROI D was selected from geometric coverage on the same annotations; ROI findings are exploratory and selection-biased.",
            "GOAL boxes are frozen detector proposals; a missing GOAL box yields no ROI proposal.",
            "This frame-level pilot does not measure full-pipeline candidate recall, Recall@K or review minutes.",
        ],
    }
    (OUT / "summary_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"device": device, "runtime_seconds": timings, "by_mode": summary,
                      "gpu_failure_before_cpu_fallback": failure}, ensure_ascii=False, indent=2))
    return 0


def execute(manifest, annotations, goals, device):
    model = YOLO(str(MODEL_PATH))
    rows: list[dict] = []
    timings = {"global_960": 0.0, "global_1536": 0.0, "roi_1536": 0.0}
    for item in manifest:
        sample_id = item["sample_id"]
        image_path = RUN / "frames_v1" / f"{sample_id}.jpg"
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(image_path)
        height, width = frame.shape[:2]
        g960, elapsed = infer(model, frame, 960, device); timings["global_960"] += elapsed
        g1536, elapsed = infer(model, frame, 1536, device); timings["global_1536"] += elapsed

        goal_dets = sorted(goals.get(sample_id, []), key=lambda r: -float(r["confidence"]))
        goal = goal_dets[0] if goal_dets else None
        roi = roi_from_goal(goal, width, height)
        crop_dets: list[dict] = []
        roi_area = 0.0
        if roi:
            x1, y1, x2, y2 = roi
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                roi_area = ((x2-x1)*(y2-y1))/(width*height)
                local, elapsed = infer(model, crop, 1536, device)
                timings["roi_1536"] += elapsed
                crop_dets = [{"box": (d["box"][0]+x1, d["box"][1]+y1,
                                      d["box"][2]+x1, d["box"][3]+y1),
                              "confidence": d["confidence"], "source": "roi"} for d in local]
        all960 = [{**d, "source": "global_960"} for d in g960]
        all1536 = [{**d, "source": "global_1536"} for d in g1536]
        hybrid = deduplicate(all960 + crop_dets)
        ann = annotations[sample_id]
        gt = tuple(float(ann[k]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")) if ann["ball_visibility"] == "BALL_VISIBLE" else None
        for mode, detections in (("global_960", all960), ("global_1536", all1536),
                                 ("roi_1536", crop_dets), ("hybrid_960_roi_1536", hybrid)):
            overlaps = [iou(gt, d["box"]) for d in detections] if gt else []
            matched = [i+1 for i, value in enumerate(overlaps) if value >= 0.5]
            rows.append({
                "sample_id": sample_id, "match_id": item["match_id"],
                "source_stratum": item["source_stratum"], "context_tags": ann.get("context_tags", ""),
                "ball_visibility": ann["ball_visibility"], "mode": mode,
                "goal_proposal_count": len(goal_dets), "roi_area_fraction": f"{roi_area:.6f}",
                "proposal_count": len(detections), "best_iou": f"{max(overlaps, default=0):.6f}" if gt else "",
                "hit_iou30": any(v >= 0.3 for v in overlaps) if gt else "",
                "hit_iou50": any(v >= 0.5 for v in overlaps) if gt else "",
                "true_rank_iou50": min(matched, default="") if gt else "",
                "false_proposals_iou50": len(detections) - int(any(v >= 0.5 for v in overlaps)) if gt else len(detections),
            })
        if (len(rows)//4) % 50 == 0:
            print(f"{len(rows)//4}/400 frames | device={device}", flush=True)
    return rows, timings


def summarize(rows):
    result = {}
    for mode in ("global_960", "global_1536", "roi_1536", "hybrid_960_roi_1536"):
        group = [r for r in rows if r["mode"] == mode]
        visible = [r for r in group if r["ball_visibility"] == "BALL_VISIBLE"]
        no_ball = [r for r in group if r["ball_visibility"] == "BALL_NOT_VISIBLE"]
        result[mode] = {
            "all_frames": len(group), "visible_frames": len(visible), "not_visible_frames": len(no_ball),
            "proposal_recall_iou30": sum(r["hit_iou30"] is True for r in visible)/max(1, len(visible)),
            "proposal_recall_iou50": sum(r["hit_iou50"] is True for r in visible)/max(1, len(visible)),
            "mean_false_proposals_per_frame": sum(int(r["false_proposals_iou50"]) for r in group)/max(1, len(group)),
            "rank1_share_among_iou50_matches": sum(r["true_rank_iou50"] == 1 for r in visible if r["true_rank_iou50"] != "")/max(1, sum(r["hit_iou50"] is True for r in visible)),
            "by_stratum": {},
        }
        for stratum in sorted({r["source_stratum"] for r in group}):
            sg = [r for r in visible if r["source_stratum"] == stratum]
            result[mode]["by_stratum"][stratum] = {
                "visible_frames": len(sg),
                "hit_iou50": sum(r["hit_iou50"] is True for r in sg),
                "proposal_recall_iou50": sum(r["hit_iou50"] is True for r in sg)/max(1, len(sg)),
            }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
