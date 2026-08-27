"""Corrige IDs de um lote já anotado sem alterar suas imagens ou coordenadas."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera IDs únicos para um lote de anotações.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    counters: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counters[row["match_id"]] += 1
        row["frame_id"] = f"{row['match_id']}_{args.prefix}_{counters[row['match_id']]:02d}"
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"linhas_corrigidas={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
