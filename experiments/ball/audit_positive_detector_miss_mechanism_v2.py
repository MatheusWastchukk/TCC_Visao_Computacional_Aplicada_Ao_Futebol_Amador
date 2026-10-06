"""Add an objective detector-proposal vs tracker-box breakdown for 127 TRAIN misses."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/ORACLE-TRACKER-BASELINE-EXP-0001/detector_miss_review_v1/positive_detector_misses_v1.csv"
OUT = ROOT / "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    source_rows = read_csv(SOURCE)
    if len(source_rows) != 127:
        raise ValueError(f"Expected the frozen 127 misses; got {len(source_rows)}")

    rows: list[dict[str, str]] = []
    mechanism_counts: Counter[str] = Counter()
    proposal_counts: Counter[str] = Counter()
    for source in source_rows:
        gt = list(map(float, json.loads(source["ball_of_play_box_json"])))
        candidate_count = int(source["candidate_count"])
        selected_json = source.get("selected_box_json", "")
        if not selected_json:
            selected_iou = ""
            tracker_status = "NO_TRACKER_SELECTED_BOX"
        else:
            selected = list(map(float, json.loads(selected_json)))
            selected_iou_value = iou(gt, selected)
            selected_iou = f"{selected_iou_value:.6f}"
            if selected_iou_value >= 0.5:
                tracker_status = "SELECTED_BOX_MATCHES_GT_UNEXPECTED"
            elif selected_iou_value > 0:
                tracker_status = "SELECTED_BOX_PARTIAL_OVERLAP_IOU_LT_050"
            else:
                tracker_status = "SELECTED_BOX_DISJOINT_FROM_GT"

        proposal_status = "NO_TOP10_PROPOSAL" if candidate_count == 0 else "TOP10_PROPOSALS_PRESENT_NO_MATCHING_GT"
        if tracker_status == "SELECTED_BOX_MATCHES_GT_UNEXPECTED":
            raise ValueError(f"A frozen miss has selected-box IoU >= 0.50: {source['sequence_id']} frame {source['sequence_frame']}")
        mechanism_counts[tracker_status] += 1
        proposal_counts[proposal_status] += 1
        rows.append({
            **source,
            "gt_box_width_px": f"{gt[2] - gt[0]:.2f}",
            "gt_box_height_px": f"{gt[3] - gt[1]:.2f}",
            "gt_box_area_px2": f"{(gt[2] - gt[0]) * (gt[3] - gt[1]):.2f}",
            "selected_tracker_box_iou": selected_iou,
            "proposal_status": proposal_status,
            "selected_tracker_box_status": tracker_status,
            "visual_cause_tags": "",
            "visual_evidence_note": "",
            "visual_review_status": "PENDING",
        })

    OUT.mkdir(parents=True)
    csv_path = OUT / "positive_miss_mechanism_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment_id": OUT.name,
        "scope": "TRAIN only; same frozen 127 positive frames from oracle tracker diagnostic",
        "source": str(SOURCE.relative_to(ROOT)),
        "miss_frames": len(rows),
        "proposal_status": dict(sorted(proposal_counts.items())),
        "selected_tracker_box_status": dict(sorted(mechanism_counts.items())),
        "distinction": "candidate_count measures whether the detector returned any top-10 proposals; selected_box_json is the tracker's chosen box. A displaced tracker box is not evidence that a matching detector proposal existed.",
        "limits": [
            "The supplied CSV stores only proposal count, not coordinates for all ten proposals; top-10 proposal recall is not recomputed here.",
            "Tracker box IoU is a spatial diagnostic, not a visual cause label.",
            "visual_cause_tags remain pending; do not infer blur, occlusion, or contrast from box size or tracker displacement.",
            "VALIDATION and TEST were not accessed.",
        ],
    }
    (OUT / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# BALL Positive Miss Mechanism Audit EXP-0002\n\n"
        "This audit keeps detector proposal availability separate from the tracker's selected box. It reuses the same frozen 127 TRAIN misses as EXP-0001 and adds selected-box IoU against the manually annotated ball.\n\n"
        "The frame-level table records two independent facts: whether any top-10 detector proposals existed, and whether the tracker's chosen box was absent, spatially disjoint, or partially overlapping. The original miss CSV does not include coordinates for each of the ten proposals, so this experiment cannot recompute candidate recall or prove whether the detector generated a near-miss proposal.\n\n"
        "Visual cause tags remain blank and pending. Tracker displacement is not, by itself, a visual-cause label. No VALIDATION/TEST data were accessed.\n\n"
        "Reproduce with `python experiments/ball/audit_positive_detector_miss_mechanism_v2.py`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
