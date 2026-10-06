"""Prepare objective measurements and a separate visual-cause queue for the 127 oracle misses."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/ORACLE-TRACKER-BASELINE-EXP-0001/detector_miss_review_v1/positive_detector_misses_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0001"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Saída já contém arquivos: {args.output}")
    rows = read_csv(args.input)
    if len(rows) != 127:
        raise ValueError(f"Esperados os 127 misses positivos congelados; obtidos {len(rows)}")

    reviewed: list[dict[str, object]] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    count_dist: Counter[int] = Counter()
    sequence_counts: Counter[str] = Counter()
    vis_dist: Counter[str] = Counter()
    for row in rows:
        x1, y1, x2, y2 = map(float, json.loads(row["ball_of_play_box_json"]))
        width, height = x2 - x1, y2 - y1
        area = width * height
        candidate_count = int(row["candidate_count"])
        widths.append(width)
        heights.append(height)
        areas.append(area)
        count_dist[candidate_count] += 1
        sequence_counts[row["sequence_id"]] += 1
        vis_dist[row["ball_visibility"]] += 1
        reviewed.append({
            **row,
            "gt_box_width_px": f"{width:.2f}",
            "gt_box_height_px": f"{height:.2f}",
            "gt_box_area_px2": f"{area:.2f}",
            "gt_box_area_lt_400px2": area < 400.0,
            "proposal_status": "NO_TOP10_PROPOSAL" if candidate_count == 0 else "TOP10_PRESENT_NO_IOU_050_MATCH",
            "visual_cause_tags": "",
            "visual_evidence_note": "",
            "visual_review_status": "PENDING",
        })

    args.output.mkdir(parents=True)
    csv_path = args.output / "positive_miss_audit_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)

    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 127 positive target frames from the frozen oracle tracker diagnostic",
        "input": str(args.input.relative_to(ROOT)),
        "positive_miss_frames": len(rows),
        "sequences": len(sequence_counts),
        "visibility": dict(sorted(vis_dist.items())),
        "gt_box_dimensions_px": {
            "width_median": statistics.median(widths),
            "height_median": statistics.median(heights),
            "area_median_px2": statistics.median(areas),
            "area_lt_400px2_count": sum(area < 400.0 for area in areas),
            "area_lt_400px2_fraction": round(sum(area < 400.0 for area in areas) / len(areas), 6),
        },
        "proposal_count_distribution": {str(k): count_dist[k] for k in sorted(count_dist)},
        "proposal_status": {
            "NO_TOP10_PROPOSAL": sum(int(row["candidate_count"]) == 0 for row in rows),
            "TOP10_PRESENT_NO_IOU_050_MATCH": sum(int(row["candidate_count"]) > 0 for row in rows),
        },
        "rule": "Area <400 px2 (20x20 square) is only a descriptive size flag, not a causal label or model gate. The original miss set is defined by no available candidate at IoU >=0.50; candidate_count distinguishes zero top-10 boxes from boxes with no compatible match.",
        "warning": "No causal visual tags were inferred automatically. Visual cause tags remain pending, may overlap, and must be grounded in the frame/video; do not interpret dimensions or proposal_count alone as cause.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# BALL Positive Miss Audit EXP-0001\n\n"
        "This artifact expands the frozen set of 127 positive detector misses from the oracle tracker diagnostic. It separates directly measurable facts from visual-cause annotation.\n\n"
        "## First-pass counts\n\n"
        f"- {len(rows)} frames across {len(sequence_counts)} positive sequences; 117 `VISIBLE`, 10 `PARTIAL`.\n"
        f"- GT ball-box median: {statistics.median(widths):.0f}x{statistics.median(heights):.0f}px; median area {statistics.median(areas):.0f}px2.\n"
        f"- {sum(area < 400.0 for area in areas)}/127 boxes are below the descriptive 400px2 (20x20) size flag. This is a size association, not a causal finding.\n"
        f"- {sum(int(row['candidate_count']) == 0 for row in rows)} frames have no top-10 detector boxes; {sum(int(row['candidate_count']) > 0 for row in rows)} have proposals, but none matched the target at IoU >=0.50.\n\n"
        "## Annotation fields\n\n"
        "`positive_miss_audit_v1.csv` includes objective box dimensions, visibility, top-10 candidate count, and blank `visual_cause_tags`/`visual_evidence_note` fields. Visual causes are intentionally pending rather than inferred from size or confidence. The `400px2` flag is a transparent descriptive cut and must not be treated as the universal definition of a small ball.\n\n"
        "## Reproduction\n\n"
        "```powershell\n.\\.venv\\Scripts\\python.exe experiments\\ball\\prepare_positive_detector_miss_audit_v1.py\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"experiment_id": args.output.name, "misses": len(rows), "small_box_flag_lt400": summary["gt_box_dimensions_px"]["area_lt_400px2_count"], "no_top10_proposal": summary["proposal_status"]["NO_TOP10_PROPOSAL"], "visual_causes": "PENDING"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
