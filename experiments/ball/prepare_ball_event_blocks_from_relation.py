"""Prepara blocos para revisar candidatos de coocorrência entre bola e gol."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from prepare_ball_event_blocks import FIELDS, format_time


def parse_time(value: str) -> float:
    """Converte um horário HH:MM:SS do manifesto em segundos."""
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return float(hours * 3600 + minutes * 60 + seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara blocos para revisão de relações bola-gol plausíveis."
    )
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    parser.add_argument("--review-label", default="RELACAO_PLAUSIVEL")
    args = parser.parse_args()

    with args.review_manifest.open(newline="", encoding="utf-8") as file:
        candidates = [
            row for row in csv.DictReader(file)
            if row["review_status"] == "complete" and row["review_label"] == args.review_label
        ]
    if not candidates:
        raise ValueError("Não há relações plausíveis concluídas no manifesto informado.")

    rows: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates, start=1):
        start_relation = parse_time(candidate["start_timestamp_hhmmss"])
        end_relation = parse_time(candidate["end_timestamp_hhmmss"])
        midpoint = (start_relation + end_relation) / 2.0
        start = max(0.0, midpoint - args.clip_seconds / 2.0)
        end = midpoint + args.clip_seconds / 2.0
        rows.append({
            "event_id": f"ball_event_v3_relation_{index:03d}",
            "source_interval_id": candidate["sequence_id"],
            "source_interval_label": "relacao_bola_gol_plausivel",
            "match_id": candidate["match_id"],
            "video_filename": candidate["video_filename"],
            "start_seconds": f"{start:.3f}",
            "end_seconds": f"{end:.3f}",
            "start_time_hhmmss": format_time(start),
            "end_time_hhmmss": format_time(end),
            "event_label": "",
            "annotation_status": "pending",
            "notes": (
                "Candidato gerado por relação espacial bola-gol; "
                "a revisão humana decide se há POSSIVEL_GOL."
            ),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"blocos={len(rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
