"""Cria o modelo de anotação de quatro pontos para as 35 estruturas de goleira."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = csv.DictReader(args.dataset.open(newline="", encoding="utf-8"))
    structures = [row for row in rows if row["goal_structure_label"] == "goal_structure"]
    fields = [
        "frame_id",
        "match_id",
        "timestamp",
        "original_label",
        "split",
        "image_path",
        "lt_x",
        "lt_y",
        "rt_x",
        "rt_y",
        "lb_x",
        "lb_y",
        "rb_x",
        "rb_y",
        "annotation_status",
        "annotator",
        "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in structures:
            writer.writerow(
                {
                    "frame_id": row["frame_id"],
                    "match_id": row["match_id"],
                    "timestamp": row["timestamp"],
                    "original_label": row["original_label"],
                    "split": row["split"],
                    "image_path": row["file_path"],
                    "annotation_status": "pending",
                }
            )
    print(f"estruturas_criadas={len(structures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
