"""Seleciona os candidatos mais fortes de mudança de placar para revisão."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FIELDS = ["review_id", "candidate_id", "match_id", "video_filename", "timestamp_seconds", "before_seconds", "after_seconds", "difference", "rank_in_match", "scoreboard_roi_json", "manual_label", "annotation_status", "notes"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara revisão dos sinais mais fortes de alteração visual do placar.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-expected-goal", type=int, default=3)
    parser.add_argument("--context-seconds", type=float, default=10.0)
    args = parser.parse_args()
    with args.candidates.open(newline="", encoding="utf-8") as file:
        candidates = list(csv.DictReader(file))
    with args.rois.open(newline="", encoding="utf-8") as file:
        rois = {row["match_id"]: row["scoreboard_roi_json"] for row in csv.DictReader(file)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["match_id"]].append(candidate)
    rows = []
    for match_id, values in sorted(grouped.items()):
        limit = int(values[0]["expected_goals_registered"]) * args.per_expected_goal
        for rank, candidate in enumerate(sorted(values, key=lambda row: float(row["difference"]), reverse=True)[:limit], 1):
            timestamp = float(candidate["timestamp_seconds"])
            rows.append({
                "review_id": f"score_review_{len(rows) + 1:03d}", "candidate_id": candidate["candidate_id"],
                "match_id": match_id, "video_filename": candidate["video_filename"], "timestamp_seconds": f"{timestamp:.3f}",
                "before_seconds": f"{max(0.0, timestamp - args.context_seconds):.3f}", "after_seconds": f"{timestamp + args.context_seconds:.3f}",
                "difference": candidate["difference"], "rank_in_match": str(rank), "scoreboard_roi_json": rois[match_id],
                "manual_label": "", "annotation_status": "pending", "notes": "G=placar mudou; N=ruído/sem mudança; U=inconclusivo.",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"revisoes={len(rows)}\ncsv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
