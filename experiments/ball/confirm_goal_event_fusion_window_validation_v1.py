"""Confirma uma configuração de fusão já selecionada, exclusivamente em VALIDATION."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirma fusão e janela fixa em VALIDATION.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-pre-seconds", type=float, default=15.0)
    parser.add_argument("--clip-post-seconds", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    candidates = read_csv(args.candidate_scores)
    catalog = read_csv(args.goal_timestamps)
    if any(row["split"] != "validation" for row in candidates):
        raise AssertionError("Os candidatos devem pertencer exclusivamente a VALIDATION.")
    if any(row["split"] != "validation" for row in catalog):
        raise AssertionError("O catálogo deve pertencer exclusivamente a VALIDATION; TEST permanece fechado.")
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        by_match[row["match_id"]].append(row)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if set(by_match) != set(goals_by_match):
        raise ValueError("Candidatos e catálogo VALIDATION não possuem as mesmas partidas.")
    goal_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for match_id, goals in sorted(goals_by_match.items()):
        selected = sorted(by_match[match_id], key=lambda row: float(row["fusion_score_goal"]), reverse=True)[:args.top_k]
        for rank, row in enumerate(selected, start=1):
            center = float(row["event_timestamp_seconds"])
            selected_rows.append({"match_id": match_id, "rank": rank, "source_interval_id": row["source_interval_id"], "event_timestamp_seconds": f"{center:.3f}", "fusion_score_goal": row["fusion_score_goal"], "clip_start_seconds": f"{max(0.0, center - args.clip_pre_seconds):.3f}", "clip_end_seconds": f"{center + args.clip_post_seconds:.3f}"})
        for goal in goals:
            matches = [rank for rank, row in enumerate(selected, start=1) if float(row["event_timestamp_seconds"]) - args.clip_pre_seconds <= goal <= float(row["event_timestamp_seconds"]) + args.clip_post_seconds]
            goal_rows.append({"match_id": match_id, "goal_timestamp_seconds": f"{goal:.3f}", "best_fusion_rank": matches[0] if matches else ""})
    covered = sum(bool(row["best_fusion_rank"]) for row in goal_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "modelo e escores já congelados; confirmação única em VALIDATION",
        "pipeline": "top 20 temporal previamente pontuado -> fusão temporal+visual -> top 12",
        "window": {"pre_seconds": args.clip_pre_seconds, "post_seconds": args.clip_post_seconds},
        "catalog_goals": len(goal_rows), "covered_goals": covered, "recall_at_k": covered / len(goal_rows),
        "review_minutes_per_match": args.top_k * (args.clip_pre_seconds + args.clip_post_seconds) / 60.0,
        "warning": "Resultado de confirmação; não usar para selecionar nova janela, retreinar ou alterar a configuração. TEST não foi acessado.",
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "selected_candidates_v1.csv", selected_rows)
    write_csv(args.output / "goal_coverage_v1.csv", goal_rows)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
