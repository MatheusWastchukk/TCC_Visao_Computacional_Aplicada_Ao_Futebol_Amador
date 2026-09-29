"""TRAIN-LOMO compression sweep with frozen temporal/visual/fusion rankings."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


LIMITS = (20, 30, 40, 50, 60, 80, 100)
WINDOWS = ((30, 10.0, 20.0), (40, 40.0 / 3.0, 80.0 / 3.0), (45, 15.0, 30.0), (60, 20.0, 40.0))
SCORES = ("rank_score", "visual_score_goal", "fusion_score_goal")
RRF_C = 60
NMS_IOU = 0.10


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        fps, frames = float(cap.get(cv2.CAP_PROP_FPS) or 0), float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frames / fps


def gap_select(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        center = float(row["center_seconds"])
        if all(abs(center - float(other["center_seconds"])) >= 45.0 for other in selected):
            selected.append(row)
    return selected


def iou(a: dict[str, object], b: dict[str, object]) -> float:
    a0, a1 = float(a["delivery_start_seconds"]), float(a["delivery_end_seconds"])
    b0, b1 = float(b["delivery_start_seconds"]), float(b["delivery_end_seconds"])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def nms_select(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for row in rows:
        if all(iou(row, other) <= NMS_IOU for other in kept):
            kept.append(row)
    return kept


def measure(method: str, selection: dict[str, list[dict[str, object]]], goals: dict[str, list[float]], k: int, window: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = {match: rows[:k] for match, rows in selection.items()}
    total_goals = sum(map(len, goals.values()))
    covered = 0
    segment_count = 0
    seconds_total = 0.0
    per_match: list[dict[str, object]] = []
    for match, rows in selected.items():
        match_covered = sum(any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in rows) for goal in goals[match])
        covered += match_covered
        merged: list[list[float]] = []
        for start, end in sorted((float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in rows):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        review_s = sum(e - s for s, e in merged)
        segment_count += len(merged)
        seconds_total += review_s
        per_match.append({"window_seconds": window, "method": method, "k": k, "match_id": match,
                          "catalog_goals": len(goals[match]), "covered_goals": match_covered,
                          "selected_candidates": len(rows), "merged_segments": len(merged),
                          "unique_review_minutes": f"{review_s / 60:.3f}"})
    return ({"window_seconds": window, "method": method, "k": k, "catalog_goals": total_goals,
             "covered_goals": covered, "recall": f"{covered / total_goals:.6f}",
             "candidates": sum(map(len, selected.values())), "merged_segments": segment_count,
             "unique_review_minutes_per_match": f"{seconds_total / 60 / len(selected):.3f}"}, per_match)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidates, fields = read_csv(args.candidate_scores)
    predictions, pred_fields = read_csv(args.rank_predictions)
    manifest, _ = read_csv(args.split_manifest)
    catalog, _ = read_csv(args.goal_timestamps)
    required = {"group_uid", "match_id", *SCORES}
    if missing := required - set(fields):
        raise ValueError(f"Candidate scores missing columns: {sorted(missing)}")
    if not catalog or any(r["split"].lower() != "train" for r in catalog):
        raise AssertionError("Catalog must contain TRAIN only.")
    if not candidates or len(candidates) != 5742 or any(r.get("split", "train").lower() != "train" for r in candidates):
        raise AssertionError("Expected only the full 5,742-group TRAIN LOMO pool.")
    if not {"model", "group_uid", "match_id", "best_timestamp_seconds"} <= set(pred_fields):
        raise ValueError("Rank predictions lack group identity/timestamp fields.")

    center_map = {(r["match_id"], r["group_uid"]): float(r["best_timestamp_seconds"])
                  for r in predictions if r["model"] == "random_forest"}
    if len(center_map) != len(candidates) or any((r["match_id"], r["group_uid"]) not in center_map for r in candidates):
        raise ValueError("TRAIN temporal prediction groups do not match the full-pool score file.")
    video_by_match = {r["match_id"]: r["video_filename"] for r in manifest if r["split"].lower() == "train"}
    duration_by_match = {m: duration(args.videos_dir / f) for m, f in video_by_match.items()}
    goals: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in candidates:
        center = center_map[(source["match_id"], source["group_uid"])]
        row: dict[str, object] = dict(source)
        row["center_seconds"] = center
        by_match[source["match_id"]].append(row)
    if set(by_match) != set(goals):
        raise ValueError("Candidate and catalog match sets differ.")

    # Frozen within-match RRF ranks are independent of clip duration.
    ranks_by_match: dict[str, list[dict[str, object]]] = {}
    for match, rows in by_match.items():
        rrf = {str(r["group_uid"]): 0.0 for r in rows}
        for field in SCORES:
            for rank, row in enumerate(sorted(rows, key=lambda r: (-float(r[field]), str(r["group_uid"]))), 1):
                rrf[str(row["group_uid"])] += 1.0 / (RRF_C + rank)
        for row in rows:
            row["rrf_score"] = rrf[str(row["group_uid"])]
        ranks_by_match[match] = rows

    comparison: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    for total, pre, post in WINDOWS:
        windows_by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
        for match, rows in ranks_by_match.items():
            duration_s = duration_by_match[match]
            for source in rows:
                center = float(source["center_seconds"])
                start, end = max(0.0, center - pre), min(duration_s, center + post)
                if end <= start:
                    continue
                windows_by_match[match].append({**source, "delivery_start_seconds": start, "delivery_end_seconds": end})

        rankers = {"temporal": "rank_score", "visual": "visual_score_goal", "fusion": "fusion_score_goal", "rrf_3source": "rrf_score"}
        selections: dict[str, dict[str, list[dict[str, object]]]] = {}
        for label, field in rankers.items():
            ordered = {m: sorted(rs, key=lambda r: (-float(r[field]), str(r["group_uid"]))) for m, rs in windows_by_match.items()}
            selections[f"{label}_gap45s"] = {m: gap_select(rs) for m, rs in ordered.items()}
        rrf_order = {m: sorted(rs, key=lambda r: (-float(r["rrf_score"]), str(r["group_uid"]))) for m, rs in windows_by_match.items()}
        selections[f"rrf_3source_nms_iou{NMS_IOU:.2f}"] = {m: nms_select(rs) for m, rs in rrf_order.items()}
        for method, selection in selections.items():
            for k in LIMITS:
                summary, per_match = measure(method, selection, goals, k, total)
                comparison.append(summary)
                per_match_rows.extend(per_match)

    args.output.mkdir(parents=True)
    write_csv(args.output / "shorter_delivery_window_curve_v1.csv", comparison)
    write_csv(args.output / "shorter_delivery_window_per_match_v1.csv", per_match_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; temporal, visual and fusion scores are LOMO; TEST not accessed",
        "candidate_rows": len(candidates), "matches": len(by_match), "catalog_goals": sum(map(len, goals.values())),
        "window_contract": {str(total): {"pre_seconds": pre, "post_seconds": post} for total, pre, post in WINDOWS},
        "selection": "rankings fixed; gap45 for temporal/visual/fusion/RRF; fixed RRF NMS IoU0.10 for NMS policy; no score refit",
        "candidate_limits": list(LIMITS),
        "warning": "TRAIN/LOMO exploratory compression sweep only; no VALIDATION or TEST used to choose a rule.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
