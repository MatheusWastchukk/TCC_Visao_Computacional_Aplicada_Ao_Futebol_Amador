"""Prepara manifesto de clipes anotados para o verificador, somente TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara manifesto TRAIN para o verificador de clipes.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, fields = read_csv(args.candidates)
    required = {"candidate_uid", "source_file", "match_id", "video_filename", "split", "ground_truth", "annotation_status", "candidate_start_seconds", "candidate_end_seconds"}
    if missing := required - set(fields):
        raise ValueError(f"Consolidado sem colunas: {sorted(missing)}")
    selected = []
    for row in rows:
        if row["split"] != "train":
            raise AssertionError("O manifesto aceita somente TRAIN; TEST permanece fechado.")
        if row["annotation_status"] != "completed" or row["ground_truth"] not in {"GOAL", "NO_GOAL"}:
            continue
        aligned = "ranker_candidate" in row["candidate_uid"]
        selected.append({
            "verifier_example_id": f"verifier_v1_{len(selected) + 1:04d}",
            "candidate_uid": row["candidate_uid"], "match_id": row["match_id"],
            "video_filename": row["video_filename"], "split": "train",
            "clip_start_seconds": row["candidate_start_seconds"], "clip_end_seconds": row["candidate_end_seconds"],
            "label": row["ground_truth"],
            "sampling_role": "timestamp_aligned_confirmation" if aligned else "candidate_review",
            "notes": "Manifesto de origem; não usar a fração de positivos alinhados por timestamp como estimativa de precisão.",
        })
    output_fields = list(selected[0])
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader(); writer.writerows(selected)
    summary = {"experiment_id": "GOAL-EVENT-VERIFIER-MANIFEST-V1", "scope": "somente TRAIN", "examples": len(selected), "labels": dict(Counter(row["label"] for row in selected)), "sampling_roles": dict(Counter(row["sampling_role"] for row in selected)), "note": "Manifesto para desenvolvimento do verificador; não é conjunto de avaliação independente."}
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
