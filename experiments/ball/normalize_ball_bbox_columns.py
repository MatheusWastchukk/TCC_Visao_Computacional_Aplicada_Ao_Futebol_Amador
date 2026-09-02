"""Converte caixas manuais em colunas numéricas para o contrato JSON dos avaliadores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BOX_COLUMNS = ("ball_x1", "ball_y1", "ball_x2", "ball_y2")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normaliza caixas manuais da bola para ball_of_play_box_json."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "ball_of_play_box_json" not in fields:
        fields.append("ball_of_play_box_json")

    converted = 0
    for row in rows:
        visibility = row.get("ball_visibility", "")
        values = [row.get(column, "") for column in BOX_COLUMNS]
        if visibility in {"VISIBLE", "PARTIAL"}:
            if not all(values):
                raise ValueError(f"Caixa ausente em frame visível/parcial: {row.get('sample_id', '')}")
            row["ball_of_play_box_json"] = json.dumps([int(value) for value in values])
            converted += 1
        else:
            row["ball_of_play_box_json"] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"linhas={len(rows)}")
    print(f"caixas_convertidas={converted}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
