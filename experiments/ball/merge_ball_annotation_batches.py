"""Combina as anotações originais e o lote adicional de treino da bola."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Combina lotes completos de anotações da bola.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for input_path in args.inputs:
        with input_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                sample_id = row["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"sample_id duplicado: {sample_id}")
                if row.get("annotation_status") != "completed":
                    raise ValueError(f"Anotação pendente: {sample_id}")
                seen.add(sample_id)
                rows.append(row)
    if not rows:
        raise ValueError("Nenhuma anotação encontrada")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"anotacoes={len(rows)}")
    print(f"train={sum(row['split'] == 'train' for row in rows)}")
    print(f"validation={sum(row['split'] == 'validation' for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
