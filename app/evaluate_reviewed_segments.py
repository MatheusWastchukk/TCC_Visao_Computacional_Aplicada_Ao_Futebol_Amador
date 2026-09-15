"""Avalia segmentos revisados contra timestamps catalogados de uma partida TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Lê um CSV UTF-8 com cabeçalho."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def evaluate(
    segments: list[dict[str, str]], annotations: list[dict[str, str]], timestamps: list[dict[str, str]]
) -> dict[str, object]:
    """Associa cada gol com timestamp concluído ao segmento que o contém."""
    annotations_by_segment = {row["segment_id"]: row for row in annotations}
    completed_goals = [
        row
        for row in timestamps
        if row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds")
    ]
    if not completed_goals:
        raise ValueError("Não há timestamps de gol concluídos para avaliar.")
    matches: list[dict[str, object]] = []
    for goal in completed_goals:
        timestamp = float(goal["event_timestamp_seconds"])
        matching_segments = [
            row
            for row in segments
            if float(row["start_seconds"]) <= timestamp <= float(row["end_seconds"])
        ]
        labels = [annotations_by_segment.get(row["segment_id"], {}).get("human_label", "") for row in matching_segments]
        matches.append(
            {
                "goal_catalog_id": goal["goal_catalog_id"],
                "goal_timestamp_seconds": timestamp,
                "covered_by_segment": bool(matching_segments),
                "confirmed_by_human": "GOAL" in labels,
                "segment_ids": [row["segment_id"] for row in matching_segments],
            }
        )
    labeled_goal_segments = [row for row in annotations if row.get("human_label") == "GOAL"]
    confirmed_segment_ids = {
        segment_id
        for match in matches
        if match["confirmed_by_human"]
        for segment_id in match["segment_ids"]
    }
    return {
        "completed_catalog_goals": len(matches),
        "covered_catalog_goals": sum(bool(match["covered_by_segment"]) for match in matches),
        "human_confirmed_catalog_goals": sum(bool(match["confirmed_by_human"]) for match in matches),
        "reviewed_goal_segments": len(labeled_goal_segments),
        "reviewed_goal_segments_associated_with_catalog": sum(
            row["segment_id"] in confirmed_segment_ids for row in labeled_goal_segments
        ),
        "goal_matches": matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia segmentos TRAIN revisados contra timestamps catalogados concluídos."
    )
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    segments = read_csv(args.segments)
    annotations = read_csv(args.annotations)
    timestamps = read_csv(args.timestamps)
    selected_timestamps = [row for row in timestamps if row.get("match_id") == args.match_id]
    if not selected_timestamps:
        raise ValueError(f"Partida ausente do catálogo de timestamps: {args.match_id}")
    if any(row.get("split") != "train" for row in selected_timestamps):
        raise AssertionError("Esta avaliação operacional aceita apenas TRAIN; TEST permanece fechado.")
    summary = evaluate(segments, annotations, selected_timestamps)
    summary.update({"match_id": args.match_id, "split": "train"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
