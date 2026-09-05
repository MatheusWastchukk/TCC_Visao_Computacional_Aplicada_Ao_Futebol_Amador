"""Acrescenta reinícios manualmente confirmados ao conjunto visual V3."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def hhmmss(seconds: float) -> str:
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Monta V3 com reinícios confirmados, deduplicados por partida e tempo.")
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--confirmed-contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    parser.add_argument("--dedupe-seconds", type=float, default=5.0)
    args = parser.parse_args()
    with args.base_annotations.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fields = list(rows[0])
    with args.confirmed_contexts.open(newline="", encoding="utf-8") as file:
        contexts = [
            row for row in csv.DictReader(file)
            if row["annotation_status"] == "completed" and row["context_decision"] == "RESTART_CONFIRMED"
        ]
    unique: list[dict[str, str]] = []
    for context in sorted(contexts, key=lambda row: (row["match_id"], float(row["restart_timestamp_seconds"]))):
        timestamp = float(context["restart_timestamp_seconds"])
        if any(
            old["match_id"] == context["match_id"]
            and abs(float(old["restart_timestamp_seconds"]) - timestamp) <= args.dedupe_seconds
            for old in unique
        ):
            continue
        unique.append(context)
    for context in unique:
        restart = float(context["restart_timestamp_seconds"])
        for offset in args.offsets:
            timestamp = max(0.0, restart + offset)
            rows.append(
                {
                    "sample_id": f"restart_v3_{len(rows) + 1:03d}",
                    "context_id": context["context_id"],
                    "event_id": context["event_id"],
                    "match_id": context["match_id"],
                    "video_filename": context["video_filename"],
                    "sample_timestamp_seconds": f"{timestamp:.3f}",
                    "sample_timestamp_hhmmss": hhmmss(timestamp),
                    "sampling_group": "REINICIO_CONFIRMADO_CATALOGO",
                    "reference_context_decision": "RESTART_CONFIRMED",
                    "restart_label": "REINICIO_CENTRO",
                    "annotation_status": "completed",
                    "notes": "Reinício confirmado manualmente; contexto de catálogo não é prova isolada de gol.",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"reinicios_confirmados={len(contexts)}")
    print(f"reinicios_unicos={len(unique)}")
    print(f"amostras_adicionadas={len(unique) * len(args.offsets)}")
    print(f"total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
