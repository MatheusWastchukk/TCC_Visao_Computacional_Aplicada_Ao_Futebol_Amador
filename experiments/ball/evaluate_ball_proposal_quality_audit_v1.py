"""Calcula métricas de propostas BALL após concluir a anotação manual dos 400 frames."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manual-review", type=Path, default=None)
    args = parser.parse_args()
    manifest = read_csv(args.run_dir / "sample_manifest_v1.csv")
    summary = json.loads((args.run_dir / "sample_summary_v1.json").read_text(encoding="utf-8"))
    review_path = args.manual_review or args.run_dir / "manual_ball_review_v1.csv"
    annotations = read_csv(review_path)
    annotation_by_id = {r["sample_id"]: r for r in annotations}
    if len(manifest) != int(summary["sample_count"]) or len(annotation_by_id) != len(manifest):
        raise ValueError("Manifest e anotações devem ter o mesmo conjunto de IDs.")
    pending = [r["sample_id"] for r in manifest if annotation_by_id.get(r["sample_id"], {}).get("annotation_state") != "complete"]
    if pending:
        print(json.dumps({"status": "incomplete", "pending_frames": len(pending), "sample_ids": pending[:12]}, ensure_ascii=False))
        return 2
    if any(r["split"] != "train" for r in manifest):
        raise AssertionError("TRAIN-only manifest expected; TEST must remain closed.")

    detections: dict[str, list[dict[str, str]]] = defaultdict(list)
    for det in read_csv(args.run_dir / "raw_detections_v1.csv"):
        if det["detector"] == "BALL":
            detections[det["sample_id"]].append(det)
    rows: list[dict[str, object]] = []
    frame_cache: dict[str, dict[str, object]] = {}
    for item in manifest:
        ann = annotation_by_id[item["sample_id"]]
        visibility = ann["ball_visibility"]
        gt = None
        if visibility == "BALL_VISIBLE":
            if not all(ann[k] for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")):
                raise ValueError(f"Visible ball requires a ground-truth box: {item['sample_id']}")
            gt = tuple(float(ann[k]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2"))
        elif visibility not in {"BALL_NOT_VISIBLE", "UNCERTAIN"}:
            raise ValueError(f"Unknown visibility for {item['sample_id']}: {visibility}")
        proposals = sorted(detections[item["sample_id"]], key=lambda d: int(d["rank_by_confidence"]))
        matches = [(int(d["rank_by_confidence"]), float(d["confidence"]),
                    iou(gt, tuple(float(d[k]) for k in ("x1", "y1", "x2", "y2"))))
                   for d in proposals] if gt else []
        match03 = [m for m in matches if m[2] >= 0.3]
        match05 = [m for m in matches if m[2] >= 0.5]
        true_rank = min((m[0] for m in match05), default="")
        false_props = ""
        if visibility == "BALL_NOT_VISIBLE":
            false_props = len(proposals)
        elif visibility == "BALL_VISIBLE":
            false_props = max(0, len(proposals) - (1 if match05 else 0))
        sample = {"sample_id": item["sample_id"], "match_id": item["match_id"], "source_stratum": item["source_stratum"],
            "source_id": item["source_id"], "sequence_id": item.get("sequence_id", ""),
            "sequence_frame_index": item.get("sequence_frame_index", ""), "timestamp_seconds": item["timestamp_seconds"],
            "ball_visibility": visibility, "context_tags": ann.get("context_tags", ""),
            "ball_proposal_count": len(proposals), "best_ball_confidence": max((float(d["confidence"]) for d in proposals), default=""),
            "best_iou": f"{max((m[2] for m in matches), default=0.0):.6f}" if gt else "",
            "proposal_recall_iou30": bool(match03) if gt else "", "proposal_recall_iou50": bool(match05) if gt else "",
            "true_ball_confidence_rank_iou50": true_rank, "false_proposals_iou50": false_props}
        rows.append(sample)
        frame_cache[item["sample_id"]] = {"visibility": visibility, "hit50": bool(match05), "hit30": bool(match03),
            "iou": max((m[2] for m in matches), default=0.0), "proposal_count": len(proposals), "false_proposals": false_props}

    strata_rows = []
    for stratum in sorted({r["source_stratum"] for r in rows}):
        group = [r for r in rows if r["source_stratum"] == stratum]
        visible = [r for r in group if r["ball_visibility"] == "BALL_VISIBLE"]
        not_visible = [r for r in group if r["ball_visibility"] == "BALL_NOT_VISIBLE"]
        evaluable = visible + not_visible
        strata_rows.append({"source_stratum": stratum, "annotated_frames": len(group), "visible_frames": len(visible),
            "not_visible_frames": len(not_visible), "uncertain_frames": len(group)-len(evaluable),
            "proposal_recall_iou30": f"{sum(bool(r['proposal_recall_iou30']) for r in visible)/len(visible):.6f}" if visible else "",
            "proposal_recall_iou50": f"{sum(bool(r['proposal_recall_iou50']) for r in visible)/len(visible):.6f}" if visible else "",
            "mean_false_proposals_per_evaluable_frame": f"{sum(int(r['false_proposals_iou50']) for r in evaluable)/len(evaluable):.6f}" if evaluable else "",
            "mean_ball_proposals_per_frame": f"{sum(int(r['ball_proposal_count']) for r in group)/len(group):.6f}"})

    sequence_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in manifest:
        if item.get("sequence_id"):
            sequence_groups[item["sequence_id"]].append(item)
    sequence_rows = []
    for seq_id, items in sorted(sequence_groups.items()):
        items.sort(key=lambda x: int(x["sequence_frame_index"]))
        visible = [x for x in items if annotation_by_id[x["sample_id"]]["ball_visibility"] == "BALL_VISIBLE"]
        successful = [x for x in visible if frame_cache[x["sample_id"]]["hit50"]]
        adjacent_pairs = []
        for a, b in zip(items, items[1:]):
            va = annotation_by_id[a["sample_id"]]["ball_visibility"]
            vb = annotation_by_id[b["sample_id"]]["ball_visibility"]
            delta = float(b["timestamp_seconds"]) - float(a["timestamp_seconds"])
            if va == vb == "BALL_VISIBLE" and 0 < delta <= 0.25:
                adjacent_pairs.append((a, b))
        pair_hits = sum(frame_cache[a["sample_id"]]["hit50"] and frame_cache[b["sample_id"]]["hit50"] for a, b in adjacent_pairs)
        all_three_visible = len(items) == 3 and len(visible) == 3
        sequence_rows.append({"sequence_id": seq_id, "source_stratum": items[0]["source_stratum"], "source_id": items[0]["source_id"],
            "sampled_frames": len(items), "visible_frames": len(visible), "frames_with_true_proposal_iou50": len(successful),
            "frame_coverage_iou50": f"{len(successful)/len(visible):.6f}" if visible else "",
            "evaluable_adjacent_pairs": len(adjacent_pairs), "pairs_both_have_true_proposal_iou50": pair_hits,
            "adjacent_pair_coverage_iou50": f"{pair_hits/len(adjacent_pairs):.6f}" if adjacent_pairs else "",
            "all_visible_frames_have_true_proposal": bool(all_three_visible and len(successful) == 3)})

    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.run_dir / "per_sample_proposal_quality_v1.csv", rows)
    write_csv(args.run_dir / "proposal_quality_by_stratum_v1.csv", strata_rows)
    write_csv(args.run_dir / "temporal_coverage_by_sequence_v1.csv", sequence_rows)
    visible = [r for r in rows if r["ball_visibility"] == "BALL_VISIBLE"]
    nonvisible = [r for r in rows if r["ball_visibility"] == "BALL_NOT_VISIBLE"]
    overall = {"annotated_frames": len(rows), "visible_frames": len(visible), "not_visible_frames": len(nonvisible),
        "uncertain_frames": len(rows)-len(visible)-len(nonvisible),
        "proposal_recall_iou30": sum(bool(r["proposal_recall_iou30"]) for r in visible)/len(visible) if visible else None,
        "proposal_recall_iou50": sum(bool(r["proposal_recall_iou50"]) for r in visible)/len(visible) if visible else None,
        "mean_false_proposals_per_evaluable_frame": sum(int(r["false_proposals_iou50"]) for r in visible+nonvisible)/len(visible+nonvisible) if visible or nonvisible else None,
        "confidence_rank1_share_among_matched_iou50": sum(str(r["true_ball_confidence_rank_iou50"]) == "1" for r in visible)/sum(bool(r["proposal_recall_iou50"]) for r in visible) if any(bool(r["proposal_recall_iou50"]) for r in visible) else None,
        "sequence_count": len(sequence_rows), "fully_covered_sequences": sum(bool(r["all_visible_frames_have_true_proposal"]) for r in sequence_rows)}
    result = {"experiment_id": args.run_dir.name, "scope": "TRAIN only; manually annotated frames; TEST closed",
        "matching": {"primary_iou_threshold": 0.5, "secondary_iou_threshold": 0.3,
            "false_proposal_rule": "all proposals when BALL_NOT_VISIBLE; for visible frames subtract at most one matching proposal at IoU>=0.5",
            "uncertain_frames": "excluded from recall and false-proposal means"},
        "overall": overall, "by_stratum": strata_rows,
        "limitations": ["Manual labels cover a stratified sample, not full videos.",
            "The 112 weak-negative controls are not human-confirmed goal negatives; ball visibility is annotated independently.",
            "Temporal pair coverage measures only 3-frame bursts sampled at approximately 5 FPS, not full-lance tracking."]}
    (args.run_dir / "proposal_quality_metrics_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "overall": overall, "by_stratum": strata_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
