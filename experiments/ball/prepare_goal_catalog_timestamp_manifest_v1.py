"""Prepara o manifesto TRAIN para registrar timestamps de gols catalogados."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cria uma fila TRAIN de gols catalogados ainda sem timestamp para medir cobertura."
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")

    with args.split_manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    fields = [
        "goal_catalog_id",
        "match_id",
        "video_filename",
        "split",
        "catalog_goal_index",
        "event_timestamp_seconds",
        "event_timestamp_hhmmss",
        "annotation_status",
        "notes",
    ]
    output_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") != "train":
            continue
        for index in range(1, int(row["goals_registered"]) + 1):
            output_rows.append(
                {
                    "goal_catalog_id": f"{row['match_id']}__catalog_goal_{index:03d}",
                    "match_id": row["match_id"],
                    "video_filename": row["video_filename"],
                    "split": "train",
                    "catalog_goal_index": str(index),
                    "event_timestamp_seconds": "",
                    "event_timestamp_hhmmss": "",
                    "annotation_status": "pending",
                    "notes": "Timestamp ausente no catálogo original; preencher apenas após revisão do vídeo TRAIN.",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Manifesto criado com {len(output_rows)} gols TRAIN pendentes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
