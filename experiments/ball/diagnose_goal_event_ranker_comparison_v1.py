"""Compara dois rankings LOMO por gol e por partida, somente em TRAIN."""
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
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        if any(abs(float(row["best_timestamp_seconds"]) - float(other["best_timestamp_seconds"])) < separation for other in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def ranks_by_goal(rows: list[dict[str, str]], goals: dict[str, list[float]], limit: int, separation: float, pre: float, post: float) -> dict[tuple[str, float], int | None]:
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_match[row["match_id"]].append(row)
    ranks: dict[tuple[str, float], int | None] = {}
    for match_id, timestamps in goals.items():
        selected = select_diverse(by_match[match_id], limit, separation)
        for goal in timestamps:
            matches = [index for index, row in enumerate(selected, start=1) if float(row["best_timestamp_seconds"]) - pre <= goal <= float(row["best_timestamp_seconds"]) + post]
            ranks[(match_id, goal)] = matches[0] if matches else None
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara rankings LOMO TRAIN por partida.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-model", default="random_forest")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    baseline = read_csv(args.baseline)
    if "model" in baseline[0]:
        baseline = [row for row in baseline if row["model"] == args.baseline_model]
    candidate = read_csv(args.candidate)
    catalog = read_csv(args.goal_timestamps)
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("O diagnóstico aceita somente TRAIN.")
    goals: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    baseline_ranks = ranks_by_goal(baseline, goals, args.limit, args.min_separation_seconds, args.clip_pre_seconds, args.clip_post_seconds)
    candidate_ranks = ranks_by_goal(candidate, goals, args.limit, args.min_separation_seconds, args.clip_pre_seconds, args.clip_post_seconds)
    rows: list[dict[str, object]] = []
    for (match_id, goal), baseline_rank in baseline_ranks.items():
        candidate_rank = candidate_ranks[(match_id, goal)]
        rows.append({"match_id": match_id, "goal_timestamp_seconds": f"{goal:.3f}", "baseline_rank": baseline_rank or "", "candidate_rank": candidate_rank or "", "rank_delta_candidate_minus_baseline": (candidate_rank if candidate_rank else args.limit + 1) - (baseline_rank if baseline_rank else args.limit + 1)})
    summary = {"scope": "somente TRAIN; comparação LOMO", "goals": len(rows), "improved_goals": sum(row["rank_delta_candidate_minus_baseline"] < 0 for row in rows), "worsened_goals": sum(row["rank_delta_candidate_minus_baseline"] > 0 for row in rows), "unchanged_goals": sum(row["rank_delta_candidate_minus_baseline"] == 0 for row in rows), "limit": args.limit}
    args.output.mkdir(parents=True)
    with (args.output / "goal_rank_comparison_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
