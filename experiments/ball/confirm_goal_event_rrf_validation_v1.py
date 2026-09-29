"""Confirma RRF congelado em VALIDATION sobre o pool visual validado existente."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


SCORE_FIELDS = ("rank_score", "visual_score_goal", "fusion_score_goal")
LIMITS = (4, 8, 12, 20, 60)
RRF_CONSTANT = 60
GAP_SECONDS = 45.0


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


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frames / fps


def gap_select(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    for row in rows:
        center = float(row["delivery_start_seconds"]) + 20.0
        if all(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) >= GAP_SECONDS for other in chosen):
            chosen.append(row)
    return chosen


def score_selection(
    method: str,
    by_match: dict[str, list[dict[str, object]]],
    goals: dict[str, list[float]],
    limit: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = {match: gap_select(rows)[:limit] for match, rows in by_match.items()}
    covered_per_match: dict[str, int] = {}
    reviewed: list[dict[str, object]] = []
    segments = 0
    seconds = 0.0
    covered = 0
    for match, rows in selected.items():
        match_covered = sum(
            any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in rows)
            for goal in goals[match]
        )
        covered_per_match[match] = match_covered
        covered += match_covered
        merged: list[list[float]] = []
        for row in rows:
            reviewed.append({"method": method, "k": limit, "match_id": match, **row})
        for start, end in sorted(
            (float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in rows
        ):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segments += len(merged)
        seconds += sum(end - start for start, end in merged)
    total = sum(map(len, goals.values()))
    summary = {
        "method": method,
        "k": limit,
        "catalog_goals": total,
        "covered_goals": covered,
        "recall": f"{covered / total:.6f}",
        "candidates": sum(map(len, selected.values())),
        "merged_segments": segments,
        "unique_review_minutes_per_match": f"{seconds / 60 / len(selected):.3f}",
    }
    return summary, [
        {"method": method, "k": limit, "match_id": match, "catalog_goals": len(goals[match]),
         "covered_goals": covered_per_match[match], "selected_candidates": len(selected[match])}
        for match in sorted(selected)
    ] + reviewed


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen three-source RRF confirmation on VALIDATION only.")
    parser.add_argument("--visual-scores", type=Path, required=True)
    parser.add_argument("--fusion-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    visual, visual_fields = read_csv(args.visual_scores)
    fusion, fusion_fields = read_csv(args.fusion_scores)
    required_common = {"event_id", "match_id", "video_filename", "split", "delivery_start_seconds", "delivery_end_seconds"}
    if missing := (required_common | {"rank_score", "visual_score_goal"}) - set(visual_fields):
        raise ValueError(f"Visual scores missing columns: {sorted(missing)}")
    if missing := (required_common | set(SCORE_FIELDS)) - set(fusion_fields):
        raise ValueError(f"Fusion scores missing columns: {sorted(missing)}")
    if not visual or not fusion or any(row["split"].lower() != "validation" for row in visual + fusion):
        raise AssertionError("Inputs must contain VALIDATION candidates only.")
    visual_by_id = {row["event_id"]: row for row in visual}
    fusion_by_id = {row["event_id"]: row for row in fusion}
    if len(visual_by_id) != len(visual) or set(visual_by_id) != set(fusion_by_id):
        raise ValueError("Visual and fusion scores do not have identical unique event IDs.")

    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"match_id", "video_filename", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Goal catalog missing columns: {sorted(missing)}")
    if not catalog or any(row["split"].lower() != "validation" for row in catalog):
        raise AssertionError("Goal timestamp catalog must contain VALIDATION only.")
    goals: dict[str, list[float]] = defaultdict(list)
    videos: dict[str, str] = {}
    for row in catalog:
        videos[row["match_id"]] = row["video_filename"]
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    durations = {match: video_duration(args.videos_dir / filename) for match, filename in videos.items()}
    candidates: list[dict[str, object]] = []
    for event_id, vrow in visual_by_id.items():
        frow = fusion_by_id[event_id]
        if vrow["match_id"] != frow["match_id"] or vrow["video_filename"] != frow["video_filename"]:
            raise ValueError(f"Source mismatch for {event_id}")
        row: dict[str, object] = dict(vrow)
        row["delivery_start_seconds"] = max(0.0, float(vrow["delivery_start_seconds"]))
        row["delivery_end_seconds"] = min(durations[vrow["match_id"]], float(vrow["delivery_end_seconds"]))
        if row["delivery_end_seconds"] <= row["delivery_start_seconds"]:
            continue
        row["fusion_score_goal"] = frow["fusion_score_goal"]
        row["visual_rank"] = 0
        row["fusion_rank"] = 0
        row["temporal_rank"] = 0
        row["rrf_score"] = 0.0
        candidates.append(row)

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_match[str(row["match_id"])].append(row)
    if set(by_match) != set(goals):
        raise ValueError("Candidate and completed-goal match sets differ.")

    rrf_by_match: dict[str, list[dict[str, object]]] = {}
    for match, rows in by_match.items():
        for field, rank_name in (("rank_score", "temporal_rank"), ("visual_score_goal", "visual_rank"), ("fusion_score_goal", "fusion_rank")):
            ranked = sorted(rows, key=lambda row: (-float(row[field]), str(row["event_id"])))
            for rank, row in enumerate(ranked, start=1):
                row[rank_name] = rank
                row["rrf_score"] = float(row["rrf_score"]) + 1.0 / (RRF_CONSTANT + rank)
        rrf_by_match[match] = sorted(rows, key=lambda row: (-float(row["rrf_score"]), str(row["event_id"])))

    rankings: dict[str, dict[str, list[dict[str, object]]]] = {
        method: {match: sorted(rows, key=lambda row: (-float(row[field]), str(row["event_id"]))) for match, rows in by_match.items()}
        for method, field in (("temporal", "rank_score"), ("visual", "visual_score_goal"), ("fusion", "fusion_score_goal"))
    }
    rankings["rrf_3source"] = rrf_by_match
    comparison: list[dict[str, object]] = []
    per_match: list[dict[str, object]] = []
    selected_detail: list[dict[str, object]] = []
    for method, ranked in rankings.items():
        for k in LIMITS:
            summary, detail = score_selection(method, ranked, goals, k)
            comparison.append(summary)
            for item in detail:
                if "selected_candidates" in item:
                    per_match.append(item)
                else:
                    selected_detail.append(item)

    args.output.mkdir(parents=True)
    write_csv(args.output / "validation_rrf_curve_v1.csv", comparison)
    write_csv(args.output / "validation_rrf_per_match_v1.csv", per_match)
    write_csv(args.output / "validation_rrf_selected_candidates_v1.csv", selected_detail)
    summary = {
        "experiment_id": args.output.name,
        "scope": "VALIDATION only; frozen temporal, visual and fusion scores; no fit or threshold adjustment; TEST not accessed",
        "candidate_pool_rows": len(candidates),
        "matches": len(by_match),
        "catalog_goals": sum(map(len, goals.values())),
        "rank_fusion": "RRF of rank_score, visual_score_goal and frozen fusion_score_goal; reciprocal rank constant 60",
        "selection": "45-second minimum separation; delivery window 20+40 seconds clipped to video duration; per-match K in [4, 8, 12, 20, 60]",
        "pool_limitation": "Input pool was previously reduced by temporal top-60 with 45-second separation; this confirms RRF only within that pool and is not a full-candidate-universe validation.",
        "warning": "Single VALIDATION confirmation. Do not retune from these metrics; TEST remains closed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
