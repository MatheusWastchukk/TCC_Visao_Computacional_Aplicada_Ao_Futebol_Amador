"""Measure proposal complementarity between the two frozen BALL holdout models."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tmp/ultralytics_ball_ensemble_config"))

from evaluate_ball_proposal_ceiling_v1 import (
    CHECKPOINTS, CONFIDENCES, ROOT, TOP_N, iou, read_rows, sha256,
)
from build_complementarity_matrix_v1 import write_csv

DEFAULT_ANNOTATIONS = ROOT / "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/holdout_train_annotations_v1.csv"
DEFAULT_DETECTIONS = ROOT / "experiments/ball/BALL-PROPOSAL-CEILING-EXP-0001/proposal_ceiling_raw_detections_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/BALL-DETECTOR-ENSEMBLE-EXP-0001"
DEDUP_IOUS = (0.50, 0.75, 1.01)
MODELS = ("baseline", "oversampled")


def read_gt(path: Path) -> list[dict[str, str]]:
    rows = [row for row in read_rows(path) if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}]
    if len(rows) != 104:
        raise ValueError(f"Expected 104 frozen holdout positives; found {len(rows)}")
    return rows


def as_box(row: dict[str, object]) -> tuple[float, float, float, float]:
    return tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def deduplicate(rows: list[dict[str, object]], dedup_iou: float) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (-float(row["confidence"]), str(row["checkpoint"]), int(row["rank_by_confidence"])))
    kept: list[dict[str, object]] = []
    for row in ordered:
        box = as_box(row)
        if all(iou(box, as_box(other)) < dedup_iou for other in kept):
            kept.append(row)
    return kept


def hits(rows: list[dict[str, object]], gt: tuple[float, float, float, float]) -> tuple[bool, bool]:
    overlaps = [iou(as_box(row), gt) for row in rows]
    return any(value >= 0.5 for value in overlaps), any(value >= 0.3 for value in overlaps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")

    gt_rows = read_gt(args.annotations)
    gt = {row["sample_id"]: tuple(float(row[key]) for key in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")) for row in gt_rows}
    detection_rows = read_rows(args.detections)
    by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for source in detection_rows:
        if source["checkpoint"] not in MODELS:
            continue
        by_key[(source["checkpoint"], source["frame_id"])].append({**source})
    if not by_key:
        raise ValueError("Raw detection file contains no rows from the frozen checkpoint pair")

    metrics: list[dict[str, object]] = []
    per_frame: list[dict[str, object]] = []
    for confidence in CONFIDENCES:
      for dedup_iou in DEDUP_IOUS:
        for top_n in TOP_N:
            sum_proposals = sum_before_dedup = sum_unique = 0
            hit50 = hit30 = 0
            only_baseline = only_oversampled = both = neither = 0
            for sample_id, truth in gt.items():
                individual: dict[str, list[dict[str, object]]] = {}
                for model in MODELS:
                    found = [row for row in by_key[(model, sample_id)] if float(row["confidence"]) >= confidence]
                    individual[model] = found[:top_n] if top_n is not None else found
                pooled = [row for model in MODELS for row in individual[model]]
                unique = deduplicate(pooled, dedup_iou)
                kept = unique[:top_n] if top_n is not None else unique
                base_hit50, base_hit30 = hits(individual["baseline"], truth)
                over_hit50, over_hit30 = hits(individual["oversampled"], truth)
                ensemble_hit50, ensemble_hit30 = hits(kept, truth)
                if base_hit50 and over_hit50:
                    both += 1
                elif base_hit50:
                    only_baseline += 1
                elif over_hit50:
                    only_oversampled += 1
                else:
                    neither += 1
                hit50 += int(ensemble_hit50)
                hit30 += int(ensemble_hit30)
                sum_before_dedup += len(pooled)
                sum_unique += len(unique)
                sum_proposals += len(kept)
                per_frame.append({
                    "confidence_threshold": confidence, "cross_model_dedup_iou": dedup_iou,
                    "top_n": top_n if top_n is not None else "all",
                    "frame_id": sample_id, "match_id": next(row["match_id"] for row in gt_rows if row["sample_id"] == sample_id),
                    "baseline_hit_iou_050": base_hit50, "oversampled_hit_iou_050": over_hit50,
                    "ensemble_hit_iou_050": ensemble_hit50, "ensemble_hit_iou_030": ensemble_hit30,
                    "pooled_proposals_before_dedup": len(pooled), "proposals_after_dedup_and_topn": len(kept),
                })
            metrics.append({
                "confidence_threshold": confidence, "cross_model_dedup_iou": dedup_iou,
                "top_n": top_n if top_n is not None else "all",
                "positive_frames": len(gt), "candidate_recall_iou_050": f"{hit50 / len(gt):.6f}",
                "hits_iou_050": hit50, "candidate_recall_iou_030": f"{hit30 / len(gt):.6f}",
                "hits_iou_030": hit30, "mean_proposals_per_frame": f"{sum_proposals / len(gt):.6f}",
                "mean_pooled_before_dedup_per_frame": f"{sum_before_dedup / len(gt):.6f}",
                "mean_unique_before_topn_per_frame": f"{sum_unique / len(gt):.6f}",
                "hit_partition_iou_050_both": both, "baseline_only": only_baseline,
                "oversampled_only": only_oversampled, "neither": neither,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "detector_ensemble_summary_v1.csv", metrics)
    write_csv(args.output / "detector_ensemble_by_frame_v1.csv", per_frame)
    summary = {
        "experiment_id": args.output.name, "scope": "TRAIN-only frozen five-match holdout; 104 visible/partial positives; TEST and VALIDATION not accessed",
        "annotations": {"path": str(args.annotations.relative_to(ROOT)), "sha256": sha256(args.annotations)},
        "raw_detections": {"path": str(args.detections.relative_to(ROOT)), "sha256": sha256(args.detections)},
        "checkpoints": [{"name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in CHECKPOINTS.items()],
        "models": list(MODELS), "confidence_thresholds": list(CONFIDENCES), "top_n": [10, 20, 50, "all"],
        "cross_model_dedup_iou_sweep": list(DEDUP_IOUS),
        "ensemble_ranking": "pooled detections sorted by raw confidence, then model name and within-model rank; cross-model boxes at or above the dedup IoU are suppressed; top-N applies after this merge. IoU 1.01 is the no-dedup union control.",
        "warning": "Confidence values may not be calibrated equally between checkpoints. This is a two-model exploratory engineering comparison on the fixed TRAIN holdout, not external validation or a promotion decision.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": len(metrics), "frames": len(per_frame), "models": len(MODELS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
