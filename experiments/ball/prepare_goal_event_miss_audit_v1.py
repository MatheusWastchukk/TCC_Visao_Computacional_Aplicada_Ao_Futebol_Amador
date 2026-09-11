"""Prepara uma fila auditável dos gols TRAIN sem cobertura por candidatos."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


OUTPUT_FIELDS = [
    "goal_catalog_id",
    "match_id",
    "goal_timestamp_seconds",
    "recommended_review_start_seconds",
    "recommended_review_end_seconds",
    "nearest_candidate_uid",
    "nearest_candidate_label",
    "nearest_candidate_start_seconds",
    "nearest_candidate_end_seconds",
    "nearest_candidate_distance_seconds",
    "annotation_status",
    "notes",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def require_fields(fields: list[str], required: set[str], label: str) -> None:
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{label} sem colunas obrigatórias: {sorted(missing)}")


def interval_distance(timestamp: float, start: float, end: float) -> float:
    if start <= timestamp <= end:
        return 0.0
    return min(abs(timestamp - start), abs(timestamp - end))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara auditoria dos gols TRAIN sem cobertura por janelas candidatas."
    )
    parser.add_argument("--goal-coverage", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-pre-seconds", type=float, default=20.0)
    parser.add_argument("--review-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.review_pre_seconds < 0 or args.review_post_seconds < 0:
        raise ValueError("As janelas de revisão devem ser não negativas.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}.")

    coverage, coverage_fields = read_csv(args.goal_coverage)
    candidates, candidate_fields = read_csv(args.candidates)
    require_fields(
        coverage_fields,
        {"goal_catalog_id", "match_id", "goal_timestamp_seconds", "evaluation_status", "covered"},
        "Cobertura de gols",
    )
    require_fields(
        candidate_fields,
        {"candidate_uid", "match_id", "split", "ground_truth", "candidate_start_seconds", "candidate_end_seconds"},
        "Candidatos consolidados",
    )
    if any(row["split"] != "train" for row in candidates):
        raise AssertionError("A auditoria aceita somente candidatos TRAIN; TEST permanece fechado.")

    candidates_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_match[candidate["match_id"]].append(candidate)

    rows: list[dict[str, str]] = []
    for goal in coverage:
        if goal["evaluation_status"] != "completed" or goal["covered"] != "False":
            continue
        timestamp = float(goal["goal_timestamp_seconds"])
        nearby = candidates_by_match[goal["match_id"]]
        nearest = min(
            nearby,
            key=lambda candidate: interval_distance(
                timestamp,
                float(candidate["candidate_start_seconds"]),
                float(candidate["candidate_end_seconds"]),
            ),
            default=None,
        )
        if nearest is None:
            nearest_fields = {field: "" for field in OUTPUT_FIELDS[5:10]}
        else:
            distance = interval_distance(
                timestamp,
                float(nearest["candidate_start_seconds"]),
                float(nearest["candidate_end_seconds"]),
            )
            nearest_fields = {
                "nearest_candidate_uid": nearest["candidate_uid"],
                "nearest_candidate_label": nearest["ground_truth"],
                "nearest_candidate_start_seconds": nearest["candidate_start_seconds"],
                "nearest_candidate_end_seconds": nearest["candidate_end_seconds"],
                "nearest_candidate_distance_seconds": f"{distance:.3f}",
            }
        rows.append(
            {
                "goal_catalog_id": goal["goal_catalog_id"],
                "match_id": goal["match_id"],
                "goal_timestamp_seconds": f"{timestamp:.3f}",
                "recommended_review_start_seconds": f"{max(0.0, timestamp - args.review_pre_seconds):.3f}",
                "recommended_review_end_seconds": f"{timestamp + args.review_post_seconds:.3f}",
                **nearest_fields,
                "annotation_status": "pending",
                "notes": "Gol catalogado sem cobertura temporal estrita; revisar o vídeo antes de ajustar a mineração.",
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "missed_goal_review_queue_v1.csv", OUTPUT_FIELDS, rows)
    by_match = Counter(row["match_id"] for row in rows)
    summary = {
        "experiment_id": "GOAL-EVENT-MISS-AUDIT-V1",
        "scope": "somente TRAIN",
        "missed_evaluable_catalog_goals": len(rows),
        "matches_with_missed_goals": len(by_match),
        "review_window_seconds": {
            "pre": args.review_pre_seconds,
            "post": args.review_post_seconds,
        },
        "missed_goals_by_match": dict(sorted(by_match.items())),
        "note": (
            "Fila de auditoria para investigar cobertura temporal. Não é conjunto de treino, "
            "não altera rótulos e não mede desempenho de detector online."
        ),
    }
    (args.output / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
