"""Diagnostica Recall@K e ranks do ranker congelado em VALIDATION."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_diverse(rows: list[dict[str, str]], limit: int, separation: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def covers(row: dict[str, str], goal_time: float, pre: float, post: float) -> bool:
    center = float(row["best_timestamp_seconds"])
    return center - pre <= goal_time <= center + post


def main() -> int:
    parser = argparse.ArgumentParser(description="Calcula curva Recall@K do ranking congelado em VALIDATION.")
    parser.add_argument("--ranked-groups", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["validation"], required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20, 30])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    limits = sorted(set(args.candidate_limits))
    if not limits or limits[0] < 1:
        raise ValueError("candidate-limits deve conter inteiros positivos.")

    groups, group_fields = read_csv(args.ranked_groups)
    required_groups = {"group_uid", "match_id", "best_timestamp_seconds", "rank_score"}
    if missing := required_groups - set(group_fields):
        raise ValueError(f"Ranking sem colunas: {sorted(missing)}")
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"goal_catalog_id", "match_id", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Catálogo sem colunas: {sorted(missing)}")
    if any(row["split"] != args.expected_split for row in catalog):
        raise AssertionError("O catálogo contém split incompatível.")
    goals = [row for row in catalog if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]]
    if not goals:
        raise ValueError("Não há timestamps completed para avaliar.")

    groups_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in groups:
        groups_by_match[row["match_id"]].append(row)
    if {goal["match_id"] for goal in goals} - set(groups_by_match):
        raise AssertionError("Há gols sem grupos do ranking correspondente.")
    selected_by_match = {
        match_id: select_diverse(rows, max(limits), args.min_separation_seconds)
        for match_id, rows in groups_by_match.items()
    }

    best_rank_rows: list[dict[str, object]] = []
    for goal in goals:
        match_id = goal["match_id"]
        goal_time = float(goal["event_timestamp_seconds"])
        raw_rows = sorted(groups_by_match[match_id], key=lambda item: float(item["rank_score"]), reverse=True)
        selected_rows = selected_by_match[match_id]
        raw_ranks = [index for index, row in enumerate(raw_rows, start=1) if covers(row, goal_time, args.clip_pre_seconds, args.clip_post_seconds)]
        diverse_ranks = [index for index, row in enumerate(selected_rows, start=1) if covers(row, goal_time, args.clip_pre_seconds, args.clip_post_seconds)]
        best_rank_rows.append({
            "goal_catalog_id": goal["goal_catalog_id"], "match_id": match_id,
            "goal_timestamp_seconds": f"{goal_time:.3f}",
            "groups_covering_goal": len(raw_ranks),
            "best_raw_group_rank": raw_ranks[0] if raw_ranks else "",
            "best_diverse_candidate_rank": diverse_ranks[0] if diverse_ranks else "",
            "covered_at_max_k": str(bool(diverse_ranks)),
        })

    recall_rows: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    for limit in limits:
        covered_goals = sum(
            int(row["best_diverse_candidate_rank"] != "" and int(row["best_diverse_candidate_rank"]) <= limit)
            for row in best_rank_rows
        )
        recall_rows.append({"k": limit, "catalog_goals": len(goals), "covered_goals": covered_goals, "recall_at_k": f"{covered_goals / len(goals):.6f}"})
        for match_id in sorted({goal["match_id"] for goal in goals}):
            match_goals = [row for row in best_rank_rows if row["match_id"] == match_id]
            match_covered = sum(
                int(row["best_diverse_candidate_rank"] != "" and int(row["best_diverse_candidate_rank"]) <= limit)
                for row in match_goals
            )
            per_match_rows.append({"match_id": match_id, "k": limit, "catalog_goals": len(match_goals), "covered_goals": match_covered, "recall_at_k": f"{match_covered / len(match_goals):.6f}"})

    args.output.mkdir(parents=True)
    write_csv(args.output / "recall_at_k_v1.csv", recall_rows, ["k", "catalog_goals", "covered_goals", "recall_at_k"])
    write_csv(args.output / "goal_best_rank_v1.csv", best_rank_rows, list(best_rank_rows[0]))
    write_csv(args.output / "per_match_recall_at_k_v1.csv", per_match_rows, ["match_id", "k", "catalog_goals", "covered_goals", "recall_at_k"])
    summary = {
        "experiment_id": args.output.name,
        "scope": "VALIDATION; diagnóstico do Random Forest TRAIN congelado",
        "candidate_limits": limits,
        "catalog_goals": len(goals),
        "min_separation_seconds": args.min_separation_seconds,
        "clip_window_seconds": {"pre": args.clip_pre_seconds, "post": args.clip_post_seconds},
        "note": "O rank diverso mede a ordem efetiva dos candidatos após a separação temporal; não é resultado de novo treinamento.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recall_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
