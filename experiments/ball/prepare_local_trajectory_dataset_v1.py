"""Acrescenta trajetória local a grupos TRAIN já rotulados no desenvolvimento.

Esta etapa só prepara uma matriz para avaliação leave-one-match-out. Não
consulta VALIDATION/TEST, não altera os arquivos de origem e não promove a
evidência ao pipeline V4.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LOCAL_FIELDS = ["path_observations", "path_score", "mean_normalized_step", "continuity"]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara o dataset TRAIN com atributos de trajetória local.")
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--local-features", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-fields", nargs="+", default=LOCAL_FIELDS)
    parser.add_argument("--feature-prefix", default="local_trajectory_")
    parser.add_argument("--dataset-filename", default="temporal_group_dataset_local_trajectory_v1.csv")
    parser.add_argument("--expected-split", choices=["train", "validation"], default="train")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    base_rows, base_fields = read_csv(args.base_dataset)
    local_rows, _ = read_csv(args.local_features)
    local_by_group = {row["group_uid"]: row for row in local_rows}
    matched = [row for row in base_rows if row["group_uid"] in local_by_group]
    if not matched:
        raise ValueError("Nenhum group_uid do dataset base foi encontrado nos atributos locais.")
    matches = {row["match_id"] for row in matched}
    enriched = [{
        **row,
        **{f"{args.feature_prefix}{field}": local_by_group[row["group_uid"]][field] for field in args.evidence_fields},
    } for row in matched]
    timestamp_rows, timestamp_fields = read_csv(args.goal_timestamps)
    timestamp_subset = [
        row for row in timestamp_rows
        if row.get("split") == args.expected_split and row.get("match_id") in matches
    ]
    if not timestamp_subset:
        raise ValueError("Não há timestamps TRAIN concluídos para as partidas presentes no dataset.")
    args.output.mkdir(parents=True)
    write_csv(args.output / "temporal_group_dataset_baseline_subset_v1.csv", matched, base_fields)
    fields = [*base_fields, *(f"{args.feature_prefix}{field}" for field in args.evidence_fields)]
    write_csv(args.output / args.dataset_filename, enriched, fields)
    write_csv(args.output / "goal_timestamps_train_subset_v1.csv", timestamp_subset, timestamp_fields)
    summary = {
        "experiment_id": args.output.name,
        "scope": f"somente {args.expected_split.upper()}; preparação da matriz experimental",
        "base_groups": len(base_rows),
        "matched_groups": len(enriched),
        "matches": len(matches),
        "experimental_features": [f"{args.feature_prefix}{field}" for field in args.evidence_fields],
        "catalog_timestamps": len(timestamp_subset),
        "warning": "A matriz preparada não é resultado de avaliação e não altera o V4.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
