"""Reúne manifestos de candidatos em uma fila única e segura de anotação."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reúne candidatos TRAIN em um CSV único para anotação humana."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fields: list[str] = []
    seen_ids: set[str] = set()
    for input_path in args.inputs:
        with input_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            source_rows = list(reader)
            source_fields = reader.fieldnames or []
        required = {"event_id", "match_id", "split", "video_filename", "event_timestamp_seconds"}
        missing = required - set(source_fields)
        if missing:
            raise ValueError(f"Manifesto inválido {input_path}: colunas ausentes {sorted(missing)}")
        if any(row.get("split") != "train" for row in source_rows):
            raise AssertionError("A fila de anotação aceita somente candidatos do split TRAIN.")
        for field in source_fields:
            if field not in fields:
                fields.append(field)
        for row in source_rows:
            batch_id = f"{row['match_id']}__{row['event_id']}"
            if batch_id in seen_ids:
                raise ValueError(f"event_id duplicado na fila: {batch_id}")
            seen_ids.add(batch_id)
            combined = dict(row)
            combined["source_event_id"] = row["event_id"]
            combined["event_id"] = batch_id
            rows.append(combined)

    if not rows:
        raise ValueError("Nenhum candidato foi encontrado nos manifestos informados.")
    if "source_event_id" not in fields:
        fields.append("source_event_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Fila criada com {len(rows)} candidatos: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
