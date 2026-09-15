"""Caracteriza o reranking por fusão em janelas de entrega, somente em TRAIN."""
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
    parser = argparse.ArgumentParser(description="Caracteriza fusão LOMO TRAIN sem novo treinamento.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    candidates = read_csv(args.candidate_scores)
    catalog = read_csv(args.goal_timestamps)
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("A caracterização aceita exclusivamente TRAIN.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_match[row["match_id"]].append(row)
    if set(by_match) != set(goals_by_match):
        raise ValueError("Candidatos de fusão e catálogo TRAIN não possuem as mesmas partidas.")
    windows = [(10.0, 20.0), (15.0, 30.0), (20.0, 40.0)]
    methods = ["rank_score", "visual_score_goal", "fusion_score_goal"]
    limits = sorted(set(args.candidate_limits))
    rows: list[dict[str, object]] = []
    all_goals = sum(len(values) for values in goals_by_match.values())
    for pre, post in windows:
        for method in methods:
            for limit in limits:
                covered = 0
                for match_id, goals in goals_by_match.items():
                    selected = sorted(by_match[match_id], key=lambda row: float(row[method]), reverse=True)[:limit]
                    for goal in goals:
                        covered += int(any(float(row["candidate_start_seconds"]) + 10.0 - pre <= goal <= float(row["candidate_start_seconds"]) + 10.0 + post for row in selected))
                rows.append({
                    "clip_pre_seconds": pre, "clip_post_seconds": post, "method": method, "k": limit,
                    "catalog_goals": all_goals, "covered_goals": covered, "recall_at_k": f"{covered / all_goals:.6f}",
                    "review_minutes_per_match": f"{limit * (pre + post) / 60.0:.2f}",
                })
    summary = {
        "experiment_id": args.output.name,
        "scope": "somente TRAIN; escores de fusão out-of-fold já existentes",
        "preselection": "top 20 do ranker temporal",
        "note": "A avaliação muda somente a janela do clipe entregue. O visual_score foi extraído da janela original 10+20 e não foi retreinado para janelas maiores.",
        "warning": "Sem acesso a VALIDATION ou TEST; resultados de janela maior são análise de desenvolvimento, não confirmação externa.",
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "fusion_window_retrieval_curve_v1.csv", rows)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
