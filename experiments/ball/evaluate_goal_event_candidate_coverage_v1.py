"""Mede cobertura de gols TRAIN por janelas de candidatos anotados."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_fields(fields: list[str], required: set[str], description: str) -> None:
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{description} sem colunas obrigatórias: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mede cobertura de gols anotados por janelas de candidatos, somente em TRAIN."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--boundary-tolerance-seconds",
        type=float,
        default=0.0,
        help="Margem temporal simétrica aplicada às bordas da janela do candidato.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.boundary_tolerance_seconds < 0:
        raise ValueError("--boundary-tolerance-seconds deve ser não negativo.")

    candidates, candidate_fields = read_csv(args.candidates)
    goals, goal_fields = read_csv(args.goal_timestamps)
    require_fields(
        candidate_fields,
        {"candidate_uid", "match_id", "split", "ground_truth", "candidate_start_seconds", "candidate_end_seconds"},
        "Candidatos",
    )
    require_fields(
        goal_fields,
        {"goal_catalog_id", "match_id", "split", "event_timestamp_seconds", "annotation_status"},
        "Timestamps de gol",
    )
    if any(row["split"] != "train" for row in candidates + goals):
        raise AssertionError("A avaliação aceita somente registros TRAIN; TEST permanece fechado.")

    candidates_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        start = float(candidate["candidate_start_seconds"])
        end = float(candidate["candidate_end_seconds"])
        if end < start:
            raise ValueError(f"Janela inválida no candidato {candidate['candidate_uid']}")
        candidates_by_match[candidate["match_id"]].append(candidate)

    coverage_rows: list[dict[str, str]] = []
    candidate_matches: dict[str, list[str]] = defaultdict(list)
    evaluable_goals = 0
    covered_goals = 0
    not_evaluable_goals = 0
    catalog_discrepancy_goals = 0
    for goal in goals:
        status = goal["annotation_status"]
        if status in {"not_evaluable", "catalog_discrepancy"}:
            if status == "not_evaluable":
                not_evaluable_goals += 1
            else:
                catalog_discrepancy_goals += 1
            coverage_rows.append(
                {
                    "goal_catalog_id": goal["goal_catalog_id"],
                    "match_id": goal["match_id"],
                    "goal_timestamp_seconds": "",
                    "evaluation_status": status,
                    "covered": "",
                    "matching_candidate_uids": "",
                    "matching_candidate_labels": "",
                }
            )
            continue
        if status != "completed" or not goal["event_timestamp_seconds"]:
            raise ValueError(f"Gol sem timestamp ou status inválido: {goal['goal_catalog_id']}")
        evaluable_goals += 1
        timestamp = float(goal["event_timestamp_seconds"])
        matches = [
            candidate
            for candidate in candidates_by_match[goal["match_id"]]
            if (
                float(candidate["candidate_start_seconds"]) - args.boundary_tolerance_seconds
                <= timestamp
                <= float(candidate["candidate_end_seconds"]) + args.boundary_tolerance_seconds
            )
        ]
        for candidate in matches:
            candidate_matches[candidate["candidate_uid"]].append(goal["goal_catalog_id"])
        covered = bool(matches)
        covered_goals += int(covered)
        coverage_rows.append(
            {
                "goal_catalog_id": goal["goal_catalog_id"],
                "match_id": goal["match_id"],
                "goal_timestamp_seconds": f"{timestamp:.3f}",
                "evaluation_status": "completed",
                "covered": str(covered),
                "matching_candidate_uids": ";".join(candidate["candidate_uid"] for candidate in matches),
                "matching_candidate_labels": ";".join(candidate["ground_truth"] for candidate in matches),
            }
        )

    candidate_rows = []
    for candidate in candidates:
        matched_goals = candidate_matches[candidate["candidate_uid"]]
        candidate_rows.append(
            {
                "candidate_uid": candidate["candidate_uid"],
                "match_id": candidate["match_id"],
                "manual_label": candidate["ground_truth"],
                "candidate_start_seconds": candidate["candidate_start_seconds"],
                "candidate_end_seconds": candidate["candidate_end_seconds"],
                "evaluation_window_start_seconds": f"{float(candidate['candidate_start_seconds']) - args.boundary_tolerance_seconds:.3f}",
                "evaluation_window_end_seconds": f"{float(candidate['candidate_end_seconds']) + args.boundary_tolerance_seconds:.3f}",
                "matched_catalog_goal_ids": ";".join(matched_goals),
                "matches_catalog_goal_window": str(bool(matched_goals)),
            }
        )

    by_match = defaultdict(lambda: Counter())
    for row in coverage_rows:
        counter = by_match[row["match_id"]]
        if row["evaluation_status"] == "not_evaluable":
            counter["not_evaluable"] += 1
        elif row["evaluation_status"] == "catalog_discrepancy":
            counter["catalog_discrepancy"] += 1
        else:
            counter["evaluable"] += 1
            counter["covered"] += int(row["covered"] == "True")
    match_rows = []
    for match_id in sorted(by_match):
        counter = by_match[match_id]
        recall = counter["covered"] / counter["evaluable"] if counter["evaluable"] else None
        match_rows.append(
            {
                "match_id": match_id,
                "evaluable_goals": counter["evaluable"],
                "covered_goals": counter["covered"],
                "not_evaluable_goals": counter["not_evaluable"],
                "catalog_discrepancy_goals": counter["catalog_discrepancy"],
                "candidate_window_recall": f"{recall:.6f}" if recall is not None else "",
            }
        )

    labels = Counter(row["manual_label"] for row in candidate_rows)
    candidate_hits = sum(row["matches_catalog_goal_window"] == "True" for row in candidate_rows)
    summary = {
        "experiment_id": "GOAL-EVENT-CANDIDATE-COVERAGE-V1",
        "scope": "somente TRAIN",
        "candidate_windows": len(candidate_rows),
        "boundary_tolerance_seconds": args.boundary_tolerance_seconds,
        "manual_labels": dict(sorted(labels.items())),
        "evaluable_catalog_goals": evaluable_goals,
        "not_evaluable_catalog_goals": not_evaluable_goals,
        "catalog_discrepancy_catalog_goals": catalog_discrepancy_goals,
        "covered_catalog_goals": covered_goals,
        "candidate_window_recall": round(covered_goals / evaluable_goals, 6) if evaluable_goals else None,
        "candidates_matching_any_catalog_goal_window": candidate_hits,
        "note": (
            "Recall mede apenas a cobertura temporal das janelas dos candidatos anotados. "
            "Não equivale à confirmação online do evento e exclui gols sem timestamp visual confiável "
            "ou com divergência confirmada entre catálogo e vídeo."
        ),
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "goal_coverage_v1.csv", list(coverage_rows[0]), coverage_rows)
    write_csv(args.output / "candidate_goal_associations_v1.csv", list(candidate_rows[0]), candidate_rows)
    write_csv(args.output / "per_match_coverage_v1.csv", list(match_rows[0]), match_rows)
    (args.output / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
