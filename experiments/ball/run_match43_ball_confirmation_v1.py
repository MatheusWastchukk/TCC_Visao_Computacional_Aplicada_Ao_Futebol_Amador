"""Run frozen BALL resolution and ROI confirmation on annotated Match 43 frames."""
from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/ball/BALL-HIGH-RES-CONFIRMATION-EXP-0001-MATCH43"
OUT = EXP / "results_v2"
CACHE = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/miopen_workspace_cache_v1"
(CACHE / "user_db").mkdir(parents=True, exist_ok=True)
(CACHE / "kernel_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(CACHE / "user_db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(CACHE / "kernel_cache"))

import cv2
import torch
from ultralytics import YOLO

ANN = EXP / "match43_ball_annotations_v1_annotated.csv"
MANIFEST = EXP / "match43_ball_annotation_manifest_v1.csv"
BALL_MODEL_PATH = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
GOAL_MODEL_PATH = ROOT / "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt"
BALL_CONF = 0.05
GOAL_CONF = 0.25
GOAL_IMGSZ = 640
BALL_SIZES = (960, 1536)
ROI_MARGIN = (2.0, 3.0, 2.0)  # lateral, above, below GOAL box dimensions
DEDUP_IOU = 0.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(1e-12, aa + ab - inter)


def predict(model: YOLO, image, conf: float, size: int, device: str, timings: dict, key: str):
    if device == "cuda:0":
        torch.cuda.synchronize()
    start = time.perf_counter()
    pred = model.predict(source=image, conf=conf, imgsz=size,
                         device=0 if device == "cuda:0" else "cpu", verbose=False)[0]
    if device == "cuda:0":
        torch.cuda.synchronize()
    timings[key] += time.perf_counter() - start
    result = []
    if pred.boxes is not None:
        for xyxy, score in zip(pred.boxes.xyxy.tolist(), pred.boxes.conf.tolist(), strict=True):
            result.append({"box": tuple(float(v) for v in xyxy), "confidence": float(score)})
    return sorted(result, key=lambda d: -d["confidence"])


def calc_roi(goal, width: int, height: int):
    if not goal:
        return None
    gx1, gy1, gx2, gy2 = goal["box"]
    gw, gh = max(1.0, gx2-gx1), max(1.0, gy2-gy1)
    lat, top, bottom = ROI_MARGIN
    return (max(0, int(gx1-lat*gw)), max(0, int(gy1-top*gh)),
            min(width, int(gx2+lat*gw)), min(height, int(gy2+bottom*gh)))


def nms_union(detections):
    kept = []
    for det in sorted(detections, key=lambda d: -d["confidence"]):
        if not any(box_iou(det["box"], prev["box"]) >= DEDUP_IOU for prev in kept):
            kept.append(det)
    return kept


def validate_inputs():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    manifest = read_csv(MANIFEST)
    annotations = read_csv(ANN)
    if len(manifest) != 400 or len(annotations) != 400:
        raise ValueError(f"Expected 400 rows in both manifests; got {len(manifest)} and {len(annotations)}")
    by_id = {row["sample_id"]: row for row in annotations}
    if set(by_id) != {row["sample_id"] for row in manifest}:
        raise ValueError("Annotation and frame manifest IDs differ")
    positives = negatives = uncertain = 0
    for item in manifest:
        ann = by_id[item["sample_id"]]
        if ann["annotation_status"] != "completed":
            raise ValueError(f"Pending annotation: {item['sample_id']}")
        if ann["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            positives += 1
            coords = [float(ann[k]) for k in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")]
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError(f"Invalid GT box: {item['sample_id']}")
        elif ann["ball_visibility"] == "NOT_VISIBLE":
            negatives += 1
        elif ann["ball_visibility"] == "UNCERTAIN":
            uncertain += 1
        else:
            raise ValueError(f"Unexpected visibility {ann['ball_visibility']} for {item['sample_id']}")
        if not Path(item["image_path"]).is_file():
            raise FileNotFoundError(item["image_path"])
    if positives == 0:
        raise ValueError("No evaluable visible-ball annotations")
    if not BALL_MODEL_PATH.is_file() or not GOAL_MODEL_PATH.is_file():
        raise FileNotFoundError("Frozen BALL or GOAL checkpoint is missing")
    return manifest, by_id, {"positive_visible_or_partial": positives, "not_visible": negatives, "uncertain_excluded": uncertain}


def execute(manifest, annotations, device):
    goal_model = YOLO(str(GOAL_MODEL_PATH))
    ball_model = YOLO(str(BALL_MODEL_PATH))
    timings = {"goal_640": 0.0, "ball_global_960": 0.0, "ball_global_1536": 0.0, "ball_roi_1536": 0.0}
    rows = []
    for index, item in enumerate(manifest, 1):
        sid = item["sample_id"]
        frame = cv2.imread(item["image_path"])
        if frame is None:
            raise RuntimeError(f"Cannot load frame {item['image_path']}")
        height, width = frame.shape[:2]
        goal_boxes = predict(goal_model, frame, GOAL_CONF, GOAL_IMGSZ, device, timings, "goal_640")
        ball960 = predict(ball_model, frame, BALL_CONF, 960, device, timings, "ball_global_960")
        ball1536 = predict(ball_model, frame, BALL_CONF, 1536, device, timings, "ball_global_1536")

        roi = calc_roi(goal_boxes[0] if goal_boxes else None, width, height)
        roi_area = 0.0
        roi_dets = []
        if roi:
            x1, y1, x2, y2 = roi
            if x2 > x1 and y2 > y1:
                roi_area = ((x2-x1)*(y2-y1))/(width*height)
                crop = frame[y1:y2, x1:x2]
                local = predict(ball_model, crop, BALL_CONF, 1536, device, timings, "ball_roi_1536")
                roi_dets = [{"box": (d["box"][0]+x1, d["box"][1]+y1,
                                      d["box"][2]+x1, d["box"][3]+y1),
                             "confidence": d["confidence"], "source": "roi"} for d in local]
        modes = {
            "global_960": ball960,
            "global_1536": ball1536,
            "roi_1536": roi_dets,
            "hybrid_960_roi_1536": nms_union(
                [{**d, "source": "global_960"} for d in ball960] + roi_dets),
        }
        ann = annotations[sid]
        visibility = ann["ball_visibility"]
        gt = tuple(float(ann[k]) for k in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")) if visibility in {"VISIBLE", "PARTIAL"} else None
        for mode, dets in modes.items():
            overlaps = [box_iou(gt, d["box"]) for d in dets] if gt else []
            matches50 = [rank for rank, value in enumerate(overlaps, 1) if value >= 0.5]
            rows.append({
                "sample_id": sid, "match_id": "match_43", "source_stratum": item["source_stratum"],
                "source_id": item["source_id"], "sequence_id": item["sequence_id"],
                "sequence_frame_index": item["sequence_frame_index"],
                "timestamp_seconds": item["timestamp_seconds"], "ball_visibility": visibility,
                "mode": mode, "goal_proposal_count": len(goal_boxes), "roi_area_fraction": f"{roi_area:.6f}",
                "proposal_count": len(dets), "best_iou": f"{max(overlaps, default=0.0):.6f}" if gt else "",
                "hit_iou30": any(v >= 0.3 for v in overlaps) if gt else "",
                "hit_iou50": any(v >= 0.5 for v in overlaps) if gt else "",
                "true_rank_iou50": min(matches50, default="") if gt else "",
                "false_proposals": len(dets)-int(any(v >= 0.5 for v in overlaps)) if gt else len(dets),
            })
        if index % 50 == 0:
            print(f"{index}/400 frames | device={device}", flush=True)
    return rows, timings


def summarize(rows):
    output = {}
    modes = ("global_960", "global_1536", "roi_1536", "hybrid_960_roi_1536")
    for mode in modes:
        group = [r for r in rows if r["mode"] == mode]
        positive = [r for r in group if r["ball_visibility"] in {"VISIBLE", "PARTIAL"}]
        all_evaluable = [r for r in group if r["ball_visibility"] != "UNCERTAIN"]
        hits50 = [r for r in positive if r["hit_iou50"] is True]
        output[mode] = {
            "annotated_frames": len(group), "visible_or_partial_frames": len(positive),
            "not_visible_frames": sum(r["ball_visibility"] == "NOT_VISIBLE" for r in group),
            "uncertain_frames_excluded": sum(r["ball_visibility"] == "UNCERTAIN" for r in group),
            "proposal_recall_iou30": sum(r["hit_iou30"] is True for r in positive)/max(1, len(positive)),
            "proposal_recall_iou50": len(hits50)/max(1, len(positive)),
            "mean_false_proposals_per_evaluable_frame": sum(int(r["false_proposals"]) for r in all_evaluable)/max(1, len(all_evaluable)),
            "rank1_share_among_iou50_matches": sum(r["true_rank_iou50"] == 1 for r in hits50)/max(1, len(hits50)),
            "by_stratum": {}, "goal_window_temporal_pairs": {},
        }
        for stratum in ("goal_window", "background_control"):
            sg = [r for r in positive if r["source_stratum"] == stratum]
            output[mode]["by_stratum"][stratum] = {
                "visible_or_partial_frames": len(sg),
                "hit_iou50": sum(r["hit_iou50"] is True for r in sg),
                "proposal_recall_iou50": sum(r["hit_iou50"] is True for r in sg)/max(1, len(sg)),
            }
        seqs = defaultdict(list)
        for r in positive:
            if r["source_stratum"] == "goal_window":
                seqs[r["sequence_id"]].append(r)
        pairs = evaluable_pairs = 0
        for seq in seqs.values():
            seq.sort(key=lambda r: int(r["sequence_frame_index"]))
            for left, right in zip(seq, seq[1:]):
                if int(right["sequence_frame_index"]) == int(left["sequence_frame_index"])+1:
                    evaluable_pairs += 1
                    pairs += int(left["hit_iou50"] is True and right["hit_iou50"] is True)
        output[mode]["goal_window_temporal_pairs"] = {
            "both_adjacent_frames_hit_iou50": pairs, "evaluable_pairs": evaluable_pairs,
            "coverage": pairs/max(1, evaluable_pairs),
        }
    return output


def main():
    manifest, annotations, label_counts = validate_inputs()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda:0" else None
    failure = None
    try:
        rows, timings = execute(manifest, annotations, device)
    except Exception as exc:
        if device == "cpu":
            raise
        failure = f"{type(exc).__name__}: {exc}"
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        device = "cpu"
        rows, timings = execute(manifest, annotations, device)

    OUT.mkdir(parents=True)
    with (OUT / "per_frame_metrics_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed", "scope": "one new external match; 400 preselected and manually annotated frames; stratified exploratory confirmation, not a multi-match estimate",
        "split": "external_confirmation", "match_id": "match_43", "device": device, "gpu": gpu if device == "cuda:0" else None,
        "torch": torch.__version__, "hip": torch.version.hip, "label_counts": label_counts,
        "models": {"ball": str(BALL_MODEL_PATH), "goal": str(GOAL_MODEL_PATH)},
        "frozen_parameters": {"ball_confidence": BALL_CONF, "goal_confidence": GOAL_CONF,
                              "goal_imgsz": GOAL_IMGSZ, "ball_global_imgsz": BALL_SIZES,
                              "roi_lateral_goal_widths": ROI_MARGIN[0], "roi_above_goal_heights": ROI_MARGIN[1],
                              "roi_below_goal_heights": ROI_MARGIN[2], "hybrid_cross_source_dedup_iou": DEDUP_IOU},
        "runtime_seconds_synchronized_inference_only": timings,
        "gpu_failure_before_cpu_fallback": failure,
        "by_mode": summarize(rows),
        "limitations": [
            "One match only; event windows are enriched around four catalogued difficult goals, with 80 stratified background controls.",
            "Ball boxes are manual; partial balls use the visible-region box and are included as positives.",
            "ROI D and its dimensions were chosen during earlier TRAIN analysis; ROI results here are a new-match diagnostic, not independently selected.",
            "This is frame proposal quality, not full pipeline candidate recall, Recall@K, or review minutes.",
            "TEST was not accessed. No training, threshold tuning, or checkpoint selection occurred.",
        ],
    }
    (OUT / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"device": device, "runtime_seconds": timings, "by_mode": summary["by_mode"],
                      "gpu_failure_before_cpu_fallback": failure}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
