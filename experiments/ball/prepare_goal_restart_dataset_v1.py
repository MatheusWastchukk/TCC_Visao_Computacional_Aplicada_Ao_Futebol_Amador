"""Prepara amostras de vídeo para anotar a evidência visual de reinício."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "sample_id", "context_id", "event_id", "match_id", "video_filename",
    "sample_timestamp_seconds", "sample_timestamp_hhmmss", "sampling_group",
    "reference_context_decision", "restart_label", "annotation_status", "notes",
]


def hhmmss(seconds: float) -> str:
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def add_sample(rows: list[dict[str, str]], context: dict[str, str], timestamp: float, group: str) -> None:
    rows.append(
        {
            "sample_id": f"restart_v1_{len(rows) + 1:03d}",
            "context_id": context["context_id"],
            "event_id": context["event_id"],
            "match_id": context["match_id"],
            "video_filename": context["video_filename"],
            "sample_timestamp_seconds": f"{max(0.0, timestamp):.3f}",
            "sample_timestamp_hhmmss": hhmmss(timestamp),
            "sampling_group": group,
            "reference_context_decision": context["context_decision"],
            "restart_label": "",
            "annotation_status": "pending",
            "notes": "Rotule somente a evidência visual deste frame; não use o resultado esperado do evento.",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara o conjunto inicial de evidência visual de reinício no centro.")
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-offsets", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    parser.add_argument("--no-restart-samples", type=int, default=3)
    args = parser.parse_args()
    if args.no_restart_samples < 1:
        raise ValueError("--no-restart-samples deve ser pelo menos 1.")
    with args.contexts.open(newline="", encoding="utf-8") as file:
        contexts = [row for row in csv.DictReader(file) if row["annotation_status"] == "completed"]

    rows: list[dict[str, str]] = []
    for context in contexts:
        if context["context_decision"] == "RESTART_CONFIRMED":
            restart = float(context["restart_timestamp_seconds"])
            for offset in args.restart_offsets:
                add_sample(rows, context, restart + offset, "ARREDOR_DO_REINICIO")
        else:
            start = float(context["search_start_seconds"])
            end = float(context["search_end_seconds"])
            for index in range(args.no_restart_samples):
                fraction = (index + 1) / (args.no_restart_samples + 1)
                add_sample(rows, context, start + ((end - start) * fraction), "CONTEXTO_SEM_REINICIO")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"amostras={len(rows)}\ncsv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
