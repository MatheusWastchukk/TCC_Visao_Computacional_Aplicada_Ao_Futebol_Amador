"""Monta um lote estratificado de negativos já revisados para avaliação de evento."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from prepare_ball_event_blocks import FIELDS, format_time
from prepare_ball_event_blocks_from_relation import parse_time
from prepare_ball_negative_events_from_relation import MOMENT_FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara lote integrado de negativos de evento.")
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--temporal-intervals", type=Path, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    parser.add_argument("--moments-output", type=Path, required=True)
    parser.add_argument("--exclude-relation-ids", nargs="*", default=[])
    parser.add_argument("--temporal-per-match", type=int, default=1)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    args = parser.parse_args()

    excluded = set(args.exclude_relation_ids)
    with args.review_manifest.open(newline="", encoding="utf-8") as file:
        relations = [
            row for row in csv.DictReader(file)
            if row["review_status"] == "complete"
            and row["review_label"] == "ASSOCIACAO_INCORRETA"
            and row["sequence_id"] not in excluded
        ]
    with args.temporal_intervals.open(newline="", encoding="utf-8") as file:
        temporal = [
            row for row in csv.DictReader(file)
            if row["annotation_status"] == "completed" and row["interval_label"] == "not_goal"
        ]
    selected_temporal: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in sorted(temporal, key=lambda value: (value["match_id"], float(value["start_seconds"]))):
        if counts.get(row["match_id"], 0) >= args.temporal_per_match:
            continue
        selected_temporal.append(row)
        counts[row["match_id"]] = counts.get(row["match_id"], 0) + 1

    candidates: list[tuple[str, str, str, str, float, str]] = []
    for row in sorted(relations, key=lambda value: (value["match_id"], value["representative_timestamp_hhmmss"])):
        candidates.append((row["sequence_id"], "relacao_bola_gol_incorreta", row["match_id"], row["video_filename"], parse_time(row["representative_timestamp_hhmmss"]), "Relação bola-gol revisada como associação incorreta."))
    for row in selected_temporal:
        midpoint = (float(row["start_seconds"]) + float(row["end_seconds"])) / 2.0
        candidates.append((row["interval_id"], "intervalo_temporal_not_goal", row["match_id"], row["video_filename"], midpoint, "Intervalo temporal anotado manualmente como not_goal."))
    if not candidates:
        raise ValueError("Nenhum negativo foi selecionado.")

    events, moments = [], []
    for index, (source_id, source_label, match_id, video, timestamp, note) in enumerate(candidates, start=1):
        start, end = max(0.0, timestamp - args.clip_seconds / 2.0), timestamp + args.clip_seconds / 2.0
        event_id = f"ball_event_v5_negative_{index:03d}"
        events.append({"event_id": event_id, "source_interval_id": source_id, "source_interval_label": source_label, "match_id": match_id, "video_filename": video, "start_seconds": f"{start:.3f}", "end_seconds": f"{end:.3f}", "start_time_hhmmss": format_time(start), "end_time_hhmmss": format_time(end), "event_label": "NAO_GOL", "annotation_status": "completed", "notes": note})
        moments.append({"event_id": event_id, "match_id": match_id, "video_filename": video, "event_label": "NAO_GOL", "event_timestamp_seconds": f"{timestamp:.3f}", "event_time_hhmmss": format_time(timestamp), "annotation_status": "completed"})
    for output, rows, fields in ((args.events_output, events, FIELDS), (args.moments_output, moments, MOMENT_FIELDS)):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    print(f"negativos={len(events)}; relacoes={len(relations)}; temporais={len(selected_temporal)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
