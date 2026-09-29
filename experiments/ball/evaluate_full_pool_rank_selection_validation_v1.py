"""Compare frozen rankers and fixed RRF/NMS selection on full VALIDATION pool."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SCORE_FIELDS = ("rank_score", "visual_score_goal", "fusion_score_goal")
LIMITS = (20, 30, 40, 50, 60, 80, 100)
GAP_SECONDS = 45.0
RRF_CONSTANT = 60
NMS_IOU = 0.10  # Fixed from the best TRAIN result; not swept on VALIDATION.


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gap_select(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    for row in rows:
        center = float(row["delivery_start_seconds"]) + 20.0
        if all(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) >= GAP_SECONDS for other in chosen):
            chosen.append(row)
    return chosen


def nms_select(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    for row in rows:
        start, end = float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])
        suppressed = False
        for other in chosen:
            other_start = float(other["delivery_start_seconds"])
            other_end = float(other["delivery_end_seconds"])
            intersection = max(0.0, min(end, other_end) - max(start, other_start))
            union = (end - start) + (other_end - other_start) - intersection
            if union > 0 and intersection / union > NMS_IOU:
                suppressed = True
                break
        if not suppressed:
            chosen.append(row)
    return chosen


def measure(method: str, selections: dict[str, list[dict[str, object]]], goals: dict[str, list[float]], k: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = {match: rows[:k] for match, rows in selections.items()}
    covered_by_match: dict[str, int] = {}
    review_seconds = 0.0
    segment_count = 0
    per_match: list[dict[str, object]] = []
    for match, rows in selected.items():
        covered = sum(
            any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in rows)
            for goal in goals[match]
        )
        covered_by_match[match] = covered
        merged: list[list[float]] = []
        for start, end in sorted((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in rows):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segment_count += len(merged)
        seconds = sum(end - start for start, end in merged)
        review_seconds += seconds
        per_match.append({"method": method, "k": k, "match_id": match, "catalog_goals": len(goals[match]),
                          "covered_goals": covered, "selected_candidates": len(rows),
                          "merged_segments": len(merged), "unique_review_minutes": f"{seconds / 60:.3f}"})
    total_goals = sum(map(len, goals.values()))
    row = {"method": method, "k": k, "catalog_goals": total_goals,
           "covered_goals": sum(covered_by_match.values()),
           "recall": f"{sum(covered_by_match.values()) / total_goals:.6f}",
           "candidates": sum(map(len, selected.values())), "merged_segments": segment_count,
           "unique_review_minutes_per_match": f"{review_seconds / 60 / len(selected):.3f}"}
    return row, per_match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidates, fields = read_csv(args.candidate_scores)
    required = {"event_id", "match_id", "split", "delivery_start_seconds", "delivery_end_seconds", *SCORE_FIELDS}
    if missing := required - set(fields):
        raise ValueError(f"Candidate scores missing columns: {sorted(missing)}")
    if not candidates or any(row["split"].lower() != "validation" for row in candidates):
        raise AssertionError("Candidate scores must contain VALIDATION only.")
    if len({row["event_id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate event_id in full VALIDATION pool.")

    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"match_id", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Goal catalog missing columns: {sorted(missing)}")
    if not catalog or any(row["split"].lower() != "validation" for row in catalog):
        raise AssertionError("Goal timestamp catalog must contain VALIDATION only.")
    goals: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in candidates:
        row: dict[str, object] = dict(source)
        row["rrf_score"] = 0.0
        by_match[row["match_id"]].append(row)
    if set(by_match) != set(goals):
        raise ValueError("Candidate and annotated-goal match sets differ.")

    rankings: dict[str, dict[str, list[dict[str, object]]]] = {
        name: {} for name in (*SCORE_FIELDS, "rrf_score")
    }
    for match, rows in by_match.items():
        for field in SCORE_FIELDS:
            ordered = sorted(rows, key=lambda row: (-float(row[field]), str(row["event_id"])))
            rankings[field][match] = ordered
            if field == SCORE_FIELDS[0]:
                for row in rows:
                    row["temporal_rank"] = 0
            rank_key = {"rank_score": "temporal_rank", "visual_score_goal": "visual_rank", "fusion_score_goal": "fusion_rank"}[field]
            for rank, row in enumerate(ordered, 1):
                row[rank_key] = rank
                row["rrf_score"] = float(row["rrf_score"]) + 1.0 / (RRF_CONSTANT + rank)
        rankings["rrf_score"][match] = sorted(rows, key=lambda row: (-float(row["rrf_score"]), str(row["event_id"])))

    selection_methods: dict[str, dict[str, list[dict[str, object]]]] = {}
    for field, label in (("rank_score", "temporal"), ("visual_score_goal", "visual"),
                         ("fusion_score_goal", "fusion"), ("rrf_score", "rrf_3source")):
        selection_methods[f"{label}_gap45s"] = {m: gap_select(rows) for m, rows in rankings[field].items()}
    selection_methods[f"rrf_3source_nms_iou{NMS_IOU:.2f}"] = {
        m: nms_select(rows) for m, rows in rankings["rrf_score"].items()
    }

    comparison: list[dict[str, object]] = []
    per_match: list[dict[str, object]] = []
    for method, selections in selection_methods.items():
        for k in LIMITS:
            summary_row, match_rows = measure(method, selections, goals, k)
            comparison.append(summary_row)
            per_match.extend(match_rows)

    args.output.mkdir(parents=True)
    write_csv(args.output / "full_pool_validation_selection_v1.csv", comparison)
    write_csv(args.output / "full_pool_validation_per_match_v1.csv", per_match)
    write_csv(args.output / "full_pool_validation_scored_candidates_v1.csv", candidates)
    summary = {
        "experiment_id": args.output.name,
        "scope": "VALIDATION only; temporal, visual, fusion frozen from TRAIN; TEST not accessed",
        "candidate_rows": len(candidates),
        "matches": len(by_match),
        "catalog_goals": sum(map(len, goals.values())),
        "rank_fusion": "RRF of temporal, visual and fusion ranks with reciprocal-rank constant 60",
        "primary_selection": "45-second center separation followed by per-match Top-K",
        "nms_selection": f"RRF-ranked windows, temporal NMS IoU={NMS_IOU:.2f}, then per-match Top-K; threshold fixed from TRAIN, not swept",
        "delivery_window": "20+40 seconds, already clipped at actual video duration",
        "candidate_limits": list(LIMITS),
        "warning": "Single full-pool VALIDATION confirmation; do not tune from these metrics. TEST remains closed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
