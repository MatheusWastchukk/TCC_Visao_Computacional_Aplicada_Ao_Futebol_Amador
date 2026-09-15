"""Diagnostica cobertura de propostas temporais antes de alterar o ranker."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_diverse(rows: list[dict[str, str]], limit: int, separation: float) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen


def covers(row: dict[str, str], goal: float, pre: float, post: float) -> bool:
    center = float(row["best_timestamp_seconds"])
    return center - pre <= goal <= center + post


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita cobertura de propostas, somente em TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--rank-limit", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    groups = read_csv(args.dataset)
    predictions = read_csv(args.rank_predictions)
    if "model" in predictions[0]:
        predictions = [row for row in predictions if row["model"] == "random_forest"]
    catalog = read_csv(args.goal_timestamps)
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("O diagnóstico aceita exclusivamente TRAIN.")
    groups_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in groups:
        groups_by_match[row["match_id"]].append(row)
    ranks_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        ranks_by_match[row["match_id"]].append(row)
    goals = [(row["match_id"], float(row["event_timestamp_seconds"])) for row in catalog if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]]
    if {match for match, _ in goals} != set(groups_by_match) or set(groups_by_match) != set(ranks_by_match):
        raise ValueError("Dataset, previsões e catálogo TRAIN não possuem as mesmas partidas.")
    windows = [(10.0, 20.0), (15.0, 30.0), (20.0, 40.0)]
    limits = [4, 8, 12, 20, 30]
    selected_by_match = {match: select_diverse(rows, args.rank_limit, args.min_separation_seconds) for match, rows in ranks_by_match.items()}
    detail_rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for pre, post in windows:
        rows_for_window: list[dict[str, object]] = []
        for match_id, goal in goals:
            all_groups = groups_by_match[match_id]
            ranked = selected_by_match[match_id]
            covering_ranks = [index for index, row in enumerate(ranked, start=1) if covers(row, goal, pre, post)]
            nearest = min(abs(float(row["best_timestamp_seconds"]) - goal) for row in all_groups)
            row = {
                "match_id": match_id,
                "goal_timestamp_seconds": f"{goal:.3f}",
                "clip_pre_seconds": pre,
                "clip_post_seconds": post,
                "any_group_covers_goal": str(any(covers(group, goal, pre, post) for group in all_groups)),
                "nearest_group_center_distance_seconds": f"{nearest:.3f}",
                "best_diverse_rank": covering_ranks[0] if covering_ranks else "",
            }
            rows_for_window.append(row)
            detail_rows.append(row)
        proposal_ceiling = sum(row["any_group_covers_goal"] == "True" for row in rows_for_window)
        for limit in limits:
            covered = sum(bool(row["best_diverse_rank"]) and int(row["best_diverse_rank"]) <= limit for row in rows_for_window)
            metrics.append({"clip_pre_seconds": pre, "clip_post_seconds": post, "k": limit, "catalog_goals": len(rows_for_window), "proposal_ceiling_all_groups": proposal_ceiling, "covered_by_ranked_candidates": covered, "recall_at_k": f"{covered / len(rows_for_window):.6f}"})
    current = [row for row in detail_rows if row["clip_pre_seconds"] == 10.0 and row["clip_post_seconds"] == 20.0]
    diagnosis = {
        "experiment_id": args.output.name,
        "scope": "somente TRAIN; diagnóstico sem retreinamento",
        "catalog_goals": len(goals),
        "current_window": {"pre_seconds": 10, "post_seconds": 20},
        "current_window_no_group": sum(row["any_group_covers_goal"] == "False" for row in current),
        "current_window_group_but_not_top30": sum(row["any_group_covers_goal"] == "True" and (not row["best_diverse_rank"] or int(row["best_diverse_rank"]) > 30) for row in current),
        "note": "Uma proposta ausente exige melhorar geração/agrupamento/janela; proposta além de top 30 exige priorização. Nenhum dado VALIDATION ou TEST foi lido.",
    }
    args.output.mkdir(parents=True)
    with (args.output / "goal_proposal_diagnosis_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0])); writer.writeheader(); writer.writerows(detail_rows)
    with (args.output / "coverage_by_window_and_rank_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    (args.output / "summary_v1.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
