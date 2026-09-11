"""Corrige IDs colididos mantendo rótulos já revisados por grupo de origem."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["match_id"], row["source_interval_id"], row["event_timestamp_seconds"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Corrige IDs de revisão de ranking já anotada.")
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--candidate-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    rows, fields = read_csv(args.annotated)
    mapping: dict[tuple[str, str, str], str] = {}
    for path in args.candidate_files:
        candidates, _ = read_csv(path)
        for candidate in candidates:
            candidate_key = key(candidate)
            if candidate_key in mapping:
                raise ValueError(f"Grupo repetido nos candidatos de origem: {candidate_key}")
            mapping[candidate_key] = candidate["event_id"]
    for row in rows:
        row_key = key(row)
        if row_key not in mapping:
            raise ValueError(f"Anotação sem candidato de origem: {row_key}")
        row["event_id"] = mapping[row_key]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Anotações reparadas: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
