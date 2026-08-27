"""Cria uma planilha inicial para selecionar frames das partidas de treino."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cria o lote inicial de seleção de frames para as partidas train."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    args = parser.parse_args()
    if args.per_category <= 0:
        raise ValueError("per-category deve ser maior que zero")

    with args.manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    train_rows = [row for row in rows if row["split"] == "train"]
    fields = [
        "sample_id",
        "match_id",
        "video_filename",
        "split",
        "sample_category",
        "timestamp_seconds",
        "label",
        "selection_status",
        "notes",
    ]
    output_rows: list[dict[str, str]] = []
    for row in train_rows:
        for category, label in [
            ("goal_structure_candidate", "pending"),
            ("hard_negative", "not_goal"),
        ]:
            for index in range(1, args.per_category + 1):
                output_rows.append(
                    {
                        "sample_id": f"{row['match_id']}_{category}_{index:02d}",
                        "match_id": row["match_id"],
                        "video_filename": row["video_filename"],
                        "split": row["split"],
                        "sample_category": category,
                        "timestamp_seconds": "",
                        "label": label,
                        "selection_status": "pending",
                        "notes": "",
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"partidas_train={len(train_rows)}")
    print(f"amostras_criadas={len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
