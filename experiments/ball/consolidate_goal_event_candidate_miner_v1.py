"""Consolida anotações do minerador de candidatos de gol sem medir recall indevido."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


VALID_LABELS = {"GOAL", "NO_GOAL", "NOT_EVALUABLE"}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolida anotações TRAIN do minerador e calcula taxa de acerto dos candidatos."
    )
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_rows, split_fields = read_csv(args.split_manifest)
    required_split = {"match_id", "split", "goals_registered"}
    missing_split = required_split - set(split_fields)
    if missing_split:
        raise ValueError(f"Manifesto de split sem colunas: {sorted(missing_split)}")
    train_catalog = {
        row["match_id"]: int(row["goals_registered"])
        for row in split_rows
        if row["split"] == "train"
    }

    consolidated: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    source_fields: list[str] = []
    for annotation_path in args.annotations:
        rows, fields = read_csv(annotation_path)
        required = {"event_id", "match_id", "split", "ground_truth", "annotation_status"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"Anotação sem colunas obrigatórias em {annotation_path}: {sorted(missing)}")
        for field in fields:
            if field not in source_fields:
                source_fields.append(field)
        for row in rows:
            if row["split"] != "train" or row["match_id"] not in train_catalog:
                raise AssertionError("A consolidação aceita somente eventos do split TRAIN.")
            if row["annotation_status"] != "completed" or row["ground_truth"] not in VALID_LABELS:
                raise ValueError(f"Anotação incompleta ou inválida: {annotation_path} / {row['event_id']}")
            key = (row["match_id"], row["event_id"])
            if key in seen:
                raise ValueError(f"Evento duplicado entre anotações: {key}")
            seen.add(key)
            item = dict(row)
            item["candidate_uid"] = f"{row['match_id']}::{row['event_id']}"
            item["source_annotation_file"] = str(annotation_path)
            consolidated.append(item)

    if not consolidated:
        raise ValueError("Nenhuma anotação foi encontrada.")
    args.output.mkdir(parents=True, exist_ok=False)
    consolidated_fields = ["candidate_uid", "source_annotation_file"] + source_fields
    write_csv(args.output / "annotated_candidates_consolidated_v1.csv", consolidated_fields, consolidated)

    by_match: dict[str, Counter[str]] = defaultdict(Counter)
    for row in consolidated:
        by_match[row["match_id"]][row["ground_truth"]] += 1
    per_match = []
    for match_id in sorted(train_catalog):
        labels = by_match[match_id]
        total = sum(labels.values())
        goals = labels["GOAL"]
        per_match.append(
            {
                "match_id": match_id,
                "cataloged_goals": train_catalog[match_id],
                "annotated_candidates": total,
                "goal_candidates": goals,
                "no_goal_candidates": labels["NO_GOAL"],
                "not_evaluable_candidates": labels["NOT_EVALUABLE"],
                "candidate_goal_rate": f"{goals / total:.6f}" if total else "",
                "has_valid_goal_candidate": str(goals > 0),
            }
        )
    write_csv(
        args.output / "per_match_summary_v1.csv",
        list(per_match[0]),
        per_match,
    )

    labels = Counter(row["ground_truth"] for row in consolidated)
    summary = {
        "experiment_id": "GOAL-EVENT-CANDIDATE-MINER-CONSOLIDATED-V1",
        "scope": "somente TRAIN",
        "annotation_files": [str(path) for path in args.annotations],
        "train_matches": len(train_catalog),
        "cataloged_goals": sum(train_catalog.values()),
        "annotated_candidates": len(consolidated),
        "labels": dict(sorted(labels.items())),
        "candidate_goal_rate": round(labels["GOAL"] / len(consolidated), 6),
        "matches_with_valid_goal_candidate": sum(row["has_valid_goal_candidate"] == "True" for row in per_match),
        "recall_available": False,
        "recall_note": (
            "Não calcular recall com este arquivo: o catálogo contém somente a contagem de gols, "
            "sem timestamps completos para associar candidatos a todos os eventos reais."
        ),
    }
    (args.output / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
