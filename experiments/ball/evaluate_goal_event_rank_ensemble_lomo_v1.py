"""Compara combinações de rankings temporais, visuais e fusion em pool TRAIN OOF."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def greedy_gap(rows: list[dict[str, str]], gap: float) -> list[dict[str, str]]:
    chosen = []
    for row in rows:
        center = (float(row["delivery_start_seconds"]) + float(row["delivery_end_seconds"])) / 2
        if all(abs(center - (float(other["delivery_start_seconds"]) + float(other["delivery_end_seconds"])) / 2) >= gap for other in chosen):
            chosen.append(row)
    return chosen


def metrics(method: str, selected: dict[str, list[dict[str, str]]], goals: dict[str, list[float]]) -> dict[str, object]:
    covered = sum(any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in selected[match])
                  for match, timestamps in goals.items() for goal in timestamps)
    segments = 0
    unique_seconds = 0.0
    for rows in selected.values():
        intervals = sorted((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in rows)
        merged: list[list[float]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segments += len(merged)
        unique_seconds += sum(end - start for start, end in merged)
    total_goals = sum(map(len, goals.values()))
    return {"method": method, "catalog_goals": total_goals, "covered_goals": covered,
            "recall": f"{covered / total_goals:.6f}", "candidates": sum(map(len, selected.values())),
            "merged_segments": segments, "unique_review_minutes_per_match": f"{unique_seconds / 60 / len(selected):.3f}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia ensemble/rank fusion com escores OOF congelados, somente TRAIN.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-temporal", type=int, default=30)
    parser.add_argument("--top-visual", type=int, default=20)
    parser.add_argument("--top-fusion", type=int, default=20)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--min-gap-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    candidates, catalog = read_csv(args.candidate_scores), read_csv(args.goal_timestamps)
    if any(row.get("split", "").lower() != "train" for row in catalog):
        raise AssertionError("O catálogo precisa ser exclusivamente TRAIN.")
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        by_match[row["match_id"]].append(row)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if set(by_match) != set(goals):
        raise ValueError("Candidatos e catálogo não contêm as mesmas partidas TRAIN.")

    rows_out: list[dict[str, object]] = []
    selections: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    score_fields = ["rank_score", "visual_score_goal", "fusion_score_goal"]
    for match, rows in by_match.items():
        ranked = {field: sorted(rows, key=lambda row: float(row[field]), reverse=True) for field in score_fields}
        selections["temporal_top60"][match] = ranked["rank_score"][:args.limit]
        rrf_score: dict[str, float] = defaultdict(float)
        for field in score_fields:
            for rank, row in enumerate(ranked[field], 1):
                rrf_score[row["candidate_id"]] += 1.0 / (60 + rank)
        rrf_rows = sorted(rows, key=lambda row: rrf_score[row["candidate_id"]], reverse=True)
        selections["RRF_temporal_visual_fusion_top60"][match] = greedy_gap(rrf_rows, args.min_gap_seconds)[:args.limit]

        union: dict[str, dict[str, str]] = {}
        union_order: list[dict[str, str]] = []
        for field, amount in [("rank_score", args.top_temporal), ("visual_score_goal", args.top_visual), ("fusion_score_goal", args.top_fusion)]:
            for row in ranked[field][:amount]:
                if row["candidate_id"] not in union:
                    union[row["candidate_id"]] = row
                    union_order.append(row)
        union_rows = union_order
        union_name = f"union_top{args.top_temporal}_temporal_{args.top_visual}_visual_{args.top_fusion}_fusion_gap{args.min_gap_seconds:g}"
        selections[union_name][match] = greedy_gap(union_rows, args.min_gap_seconds)[:args.limit]

        for key, field in [("visual_top60", "visual_score_goal"), ("fusion_top60", "fusion_score_goal")]:
            selections[key][match] = ranked[field][:args.limit]

    for method, selected in selections.items():
        rows_out.append(metrics(method, selected, goals))
    args.output.mkdir(parents=True)
    write_csv(args.output / "rank_ensemble_comparison_v1.csv", rows_out)
    summary = {"experiment_id": args.output.name, "scope": "somente TRAIN; escores por partida OOF; sem novo treinamento; TEST não acessado",
               "candidate_rows": len(candidates), "matches": len(by_match),
               "candidate_limits": {"temporal": args.top_temporal, "visual": args.top_visual, "fusion": args.top_fusion,
                                    "final_limit": args.limit, "dedup_gap_seconds": args.min_gap_seconds},
               "score_sources": str(args.candidate_scores),
               "warning": "Pool limitado aos top 60 temporais por partida; não mede recuperação fora desse pre-pool. Escores visuais/fusion são OOF de TRAIN."}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows_out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
