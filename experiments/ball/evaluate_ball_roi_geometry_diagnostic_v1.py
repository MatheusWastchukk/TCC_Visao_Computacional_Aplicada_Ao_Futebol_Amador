"""Measure how enlarged GOAL boxes cover manual BALL boxes; no inference is run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2"
OUT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/roi_geometry_diagnostic_v1"
ROI_CONFIGS = {
    "roi_a_lateral0.75_top1_bottom1": (0.75, 1.0, 1.0),
    "roi_b_lateral1_top1.5_bottom1": (1.0, 1.5, 1.0),
    "roi_c_lateral1.5_top2_bottom1.5": (1.5, 2.0, 1.5),
    "roi_d_lateral2_top3_bottom2": (2.0, 3.0, 2.0),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    manifest = read_csv(RUN / "sample_manifest_v1.csv")
    annotations = {row["sample_id"]: row for row in read_csv(RUN / "manual_ball_review_v1.csv")}
    if len(manifest) != 400 or any(row["split"] != "train" for row in manifest):
        raise ValueError("Expected the existing 400-frame TRAIN manifest only.")
    if len(annotations) != 400 or any(row["annotation_state"] != "complete" for row in annotations.values()):
        raise ValueError("Expected 400 complete manual annotations.")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {OUT}")

    goals: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(RUN / "raw_detections_v1.csv"):
        if row["detector"] == "GOAL":
            goals.setdefault(row["sample_id"], []).append(row)
    rows: list[dict[str, object]] = []
    for item in manifest:
        ann = annotations[item["sample_id"]]
        if ann["ball_visibility"] != "BALL_VISIBLE":
            continue
        gt = tuple(float(ann[k]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2"))
        frame_path = RUN / "frames_v1" / f"{item['sample_id']}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(frame_path)
        height, width = frame.shape[:2]
        predictions = sorted(goals.get(item["sample_id"], []), key=lambda row: -float(row["confidence"]))
        goal = predictions[0] if predictions else None
        goal_box = tuple(float(goal[k]) for k in ("x1", "y1", "x2", "y2")) if goal else None
        for config, (lat, top, bottom) in ROI_CONFIGS.items():
            if goal_box:
                gx1, gy1, gx2, gy2 = goal_box
                gw, gh = max(1.0, gx2 - gx1), max(1.0, gy2 - gy1)
                roi = (max(0.0, gx1-lat*gw), max(0.0, gy1-top*gh),
                       min(float(width), gx2+lat*gw), min(float(height), gy2+bottom*gh))
                roi_area_fraction = max(0.0, roi[2]-roi[0]) * max(0.0, roi[3]-roi[1]) / (width*height)
                center_x, center_y = (gt[0]+gt[2])/2, (gt[1]+gt[3])/2
                center_inside = roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]
                box_inside = roi[0] <= gt[0] and roi[1] <= gt[1] and roi[2] >= gt[2] and roi[3] >= gt[3]
            else:
                roi = (0.0, 0.0, 0.0, 0.0)
                roi_area_fraction, center_inside, box_inside = 0.0, False, False
            rows.append({
                "sample_id": item["sample_id"], "match_id": item["match_id"],
                "source_stratum": item["source_stratum"], "context_tags": ann.get("context_tags", ""),
                "roi_config": config, "goal_proposal_count": len(predictions),
                "top_goal_confidence": f"{float(goal['confidence']):.7f}" if goal else "",
                "frame_width": width, "frame_height": height,
                "roi_x1": round(roi[0], 2), "roi_y1": round(roi[1], 2),
                "roi_x2": round(roi[2], 2), "roi_y2": round(roi[3], 2),
                "roi_area_fraction": f"{roi_area_fraction:.6f}",
                "ball_center_inside": center_inside, "ball_gt_box_fully_inside": box_inside,
            })

    summaries = []
    for config in ROI_CONFIGS:
        group = [row for row in rows if row["roi_config"] == config]
        goal_found = [row for row in group if int(row["goal_proposal_count"]) > 0]
        for stratum in sorted({str(row["source_stratum"]) for row in group}):
            sg = [row for row in group if row["source_stratum"] == stratum]
            summaries.append({
                "roi_config": config, "source_stratum": stratum, "visible_ball_frames": len(sg),
                "frames_with_goal_proposal": sum(int(row["goal_proposal_count"]) > 0 for row in sg),
                "ball_center_covered": sum(bool(row["ball_center_inside"]) for row in sg),
                "ball_gt_box_fully_covered": sum(bool(row["ball_gt_box_fully_inside"]) for row in sg),
                "center_coverage_rate": sum(bool(row["ball_center_inside"]) for row in sg) / max(1, len(sg)),
                "full_box_coverage_rate": sum(bool(row["ball_gt_box_fully_inside"]) for row in sg) / max(1, len(sg)),
                "mean_roi_area_fraction": sum(float(row["roi_area_fraction"]) for row in sg) / max(1, len(sg)),
                "mean_roi_area_when_goal_found": sum(float(row["roi_area_fraction"]) for row in sg if int(row["goal_proposal_count"]) > 0) / max(1, sum(int(row["goal_proposal_count"]) > 0 for row in sg)),
            })
    OUT.mkdir(parents=True)
    write_csv(OUT / "roi_geometry_per_frame_v1.csv", rows)
    write_csv(OUT / "roi_geometry_by_stratum_v1.csv", summaries)
    overall = {}
    for config in ROI_CONFIGS:
        group = [row for row in rows if row["roi_config"] == config]
        overall[config] = {
            "visible_ball_frames": len(group),
            "frames_with_goal_proposal": sum(int(row["goal_proposal_count"]) > 0 for row in group),
            "ball_center_covered": sum(bool(row["ball_center_inside"]) for row in group),
            "ball_gt_box_fully_covered": sum(bool(row["ball_gt_box_fully_inside"]) for row in group),
            "center_coverage_rate": sum(bool(row["ball_center_inside"]) for row in group) / max(1, len(group)),
            "full_box_coverage_rate": sum(bool(row["ball_gt_box_fully_inside"]) for row in group) / max(1, len(group)),
            "mean_roi_area_fraction": sum(float(row["roi_area_fraction"]) for row in group) / max(1, len(group)),
        }
    result = {
        "status": "completed",
        "scope": "Exploratory geometric coverage only; reuses 298 visible BALL annotations from TRAIN; no inference; VALIDATION and TEST closed",
        "goal_proposals": "Top confidence GOAL box from frozen run_v2, confidence 0.25, imgsz 640",
        "roi_definitions": {name: {"lateral_margin_goal_widths": x[0], "top_margin_goal_heights": x[1], "bottom_margin_goal_heights": x[2]} for name, x in ROI_CONFIGS.items()},
        "overall": overall,
        "limitations": [
            "Coverage means manual BALL center/box containment by an expanded GOAL box; it is not ROI BALL detector recall.",
            "The sample is biased toward goal windows and weak-negative controls and reuses prior annotations.",
            "ROI unions and crop detector inference are not evaluated here.",
        ],
    }
    (OUT / "roi_geometry_summary_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
