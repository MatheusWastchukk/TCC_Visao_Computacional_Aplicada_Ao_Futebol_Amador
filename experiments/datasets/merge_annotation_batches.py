"""Consolida lotes de anotação manual sem misturar partidas entre splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolida CSVs de anotação da estrutura do gol.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for input_path in args.inputs:
        with input_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["frame_id"] in seen:
                    raise ValueError(f"frame_id duplicado: {row['frame_id']}")
                if row["split"] != args.split:
                    raise ValueError(f"Anotação fora de {args.split}: {row['frame_id']}")
                seen.add(row["frame_id"])
                rows.append(row)

    if not rows:
        raise ValueError("Nenhuma anotação encontrada")
    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "version": f"{args.split}_goal_structure_annotations_v1",
        "input_files": [str(path) for path in args.inputs],
        "total_rows": len(rows),
        "annotation_status": dict(Counter(row["annotation_status"] for row in rows)),
        "labels": dict(Counter(row["label"] for row in rows)),
        "matches": len({row["match_id"] for row in rows}),
        "status": "valid" if all(row["annotation_status"] == "completed" for row in rows) else "pending",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={summary['status']}")
    print(f"anotacoes={len(rows)}")
    print(f"partidas={summary['matches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
