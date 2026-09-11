"""Filtra candidatos TRAIN cujas janelas incluem um timestamp de gol catalogado."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara revisão dos candidatos TRAIN alinhados a timestamps.")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")

    goals, goal_fields = read_csv(args.goal_timestamps)
    required_goals = {"goal_catalog_id", "match_id", "split", "event_timestamp_seconds", "annotation_status"}
    if missing := required_goals - set(goal_fields):
        raise ValueError(f"Timestamps sem colunas: {sorted(missing)}")
    goals_by_match: dict[str, list[dict[str, str]]] = {}
    for goal in goals:
        if goal["split"] != "train":
            raise AssertionError("A revisão aceita somente TRAIN; TEST permanece fechado.")
        if goal["annotation_status"] == "completed" and goal["event_timestamp_seconds"]:
            goals_by_match.setdefault(goal["match_id"], []).append(goal)

    candidates: list[dict[str, str]] = []
    fields: list[str] | None = None
    for path in args.candidates:
        rows, current_fields = read_csv(path)
        required = {"event_id", "match_id", "split", "candidate_start_seconds", "candidate_end_seconds"}
        if missing := required - set(current_fields):
            raise ValueError(f"Candidatos sem colunas em {path}: {sorted(missing)}")
        if any(row["split"] != "train" for row in rows):
            raise AssertionError("A revisão aceita somente candidatos TRAIN; TEST permanece fechado.")
        if fields is None:
            fields = current_fields
        elif fields != current_fields:
            raise ValueError("Arquivos de candidatos possuem esquemas diferentes.")
        candidates.extend(rows)

    assert fields is not None
    output_fields = fields + ["aligned_goal_catalog_ids"]
    selected: list[dict[str, str]] = []
    for candidate in candidates:
        start = float(candidate["candidate_start_seconds"])
        end = float(candidate["candidate_end_seconds"])
        matched = [
            goal["goal_catalog_id"]
            for goal in goals_by_match.get(candidate["match_id"], [])
            if start <= float(goal["event_timestamp_seconds"]) <= end
        ]
        if matched:
            candidate["aligned_goal_catalog_ids"] = ";".join(matched)
            candidate["notes"] = (
                "Alinhado automaticamente a timestamp catalogado: " + candidate["aligned_goal_catalog_ids"] + "."
            )
            selected.append(candidate)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(f"Candidatos alinhados preparados: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
