"""Seleciona blocos curtos para anotação manual de eventos reais no vídeo."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "event_id", "source_interval_id", "source_interval_label", "match_id", "video_filename",
    "start_seconds", "end_seconds", "start_time_hhmmss", "end_time_hhmmss", "event_label",
    "annotation_status", "notes",
]


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def round_robin(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Prioriza variedade de partida ao escolher blocos de uma mesma estratificação."""
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: (value["match_id"], float(value["start_seconds"]))):
        by_match[row["match_id"]].append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        added = False
        for match_id in sorted(by_match):
            if by_match[match_id] and len(selected) < count:
                selected.append(by_match[match_id].pop(0))
                added = True
        if not added:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara blocos de vídeo para rotular eventos reais de gol.")
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source-label", type=int, default=6)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    args = parser.parse_args()
    with args.intervals.open(newline="", encoding="utf-8") as file:
        intervals = [row for row in csv.DictReader(file) if row["annotation_status"] == "completed"]
    selected: list[dict[str, str]] = []
    for source_label in ("goal_structure", "not_goal"):
        selected.extend(round_robin([row for row in intervals if row["interval_label"] == source_label], args.per_source_label))
    if not selected:
        raise ValueError("Não há intervalos concluídos com as estratificações esperadas.")
    output_rows: list[dict[str, str]] = []
    for index, interval in enumerate(selected, start=1):
        midpoint = (float(interval["start_seconds"]) + float(interval["end_seconds"])) / 2.0
        start = max(0.0, midpoint - args.clip_seconds / 2.0)
        end = midpoint + args.clip_seconds / 2.0
        output_rows.append({
            "event_id": f"ball_event_v1_{index:03d}", "source_interval_id": interval["interval_id"],
            "source_interval_label": interval["interval_label"], "match_id": interval["match_id"],
            "video_filename": interval["video_filename"], "start_seconds": f"{start:.3f}", "end_seconds": f"{end:.3f}",
            "start_time_hhmmss": format_time(start), "end_time_hhmmss": format_time(end), "event_label": "",
            "annotation_status": "pending", "notes": "Fonte temporal usada somente para estratificação; evento real exige revisão humana.",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"blocos={len(output_rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
