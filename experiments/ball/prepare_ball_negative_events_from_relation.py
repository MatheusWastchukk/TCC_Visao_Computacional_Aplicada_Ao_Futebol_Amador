"""Prepara negativos a partir de relações bola-gol já revisadas manualmente."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from prepare_ball_event_blocks import FIELDS, format_time
from prepare_ball_event_blocks_from_relation import parse_time


MOMENT_FIELDS = [
    "event_id", "match_id", "video_filename", "event_label", "event_timestamp_seconds",
    "event_time_hhmmss", "annotation_status",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara negativos por partida de relações bola-gol incorretas revisadas."
    )
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    parser.add_argument("--moments-output", type=Path, required=True)
    parser.add_argument("--per-match", type=int, default=1)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    args = parser.parse_args()

    with args.review_manifest.open(newline="", encoding="utf-8") as file:
        reviewed = [
            row for row in csv.DictReader(file)
            if row["review_status"] == "complete" and row["review_label"] == "ASSOCIACAO_INCORRETA"
        ]
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in sorted(reviewed, key=lambda value: (value["match_id"], value["representative_timestamp_hhmmss"])):
        match_id = row["match_id"]
        if counts.get(match_id, 0) >= args.per_match:
            continue
        selected.append(row)
        counts[match_id] = counts.get(match_id, 0) + 1
    if not selected:
        raise ValueError("Não há associações incorretas concluídas para preparar.")

    events: list[dict[str, str]] = []
    moments: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        timestamp = parse_time(row["representative_timestamp_hhmmss"])
        start = max(0.0, timestamp - args.clip_seconds / 2.0)
        end = timestamp + args.clip_seconds / 2.0
        event_id = f"ball_event_v4_negative_{index:03d}"
        events.append({
            "event_id": event_id, "source_interval_id": row["sequence_id"],
            "source_interval_label": "relacao_bola_gol_incorreta", "match_id": row["match_id"],
            "video_filename": row["video_filename"], "start_seconds": f"{start:.3f}", "end_seconds": f"{end:.3f}",
            "start_time_hhmmss": format_time(start), "end_time_hhmmss": format_time(end),
            "event_label": "NAO_GOL", "annotation_status": "completed",
            "notes": "Negativo derivado de relação bola-gol classificada manualmente como associação incorreta.",
        })
        moments.append({
            "event_id": event_id, "match_id": row["match_id"], "video_filename": row["video_filename"],
            "event_label": "NAO_GOL", "event_timestamp_seconds": f"{timestamp:.3f}",
            "event_time_hhmmss": format_time(timestamp), "annotation_status": "completed",
        })

    for output, rows, fields in (
        (args.events_output, events, FIELDS), (args.moments_output, moments, MOMENT_FIELDS)
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"negativos={len(events)}")
    print(f"partidas={','.join(sorted(counts))}")
    print(f"events={args.events_output}")
    print(f"moments={args.moments_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
