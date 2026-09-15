"""Mede a redução do pool top 60 com escores já congelados, somente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia redução do pool amplo, somente TRAIN.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["train", "validation"], default="train")
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[12, 20, 30, 40, 50, 60])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    candidates = read_csv(args.candidate_scores)
    catalog = read_csv(args.goal_timestamps)
    if any(row["split"] != args.expected_split for row in catalog):
        raise AssertionError("O catálogo da avaliação contém split incompatível.")
    if {row["match_id"] for row in candidates} - {row["match_id"] for row in catalog}:
        raise AssertionError("Os candidatos contêm partidas fora do catálogo esperado.")
    methods = ["rank_score", "visual_score_goal", "fusion_score_goal"]
    if any(method not in candidates[0] for method in methods):
        raise ValueError("Faltam escores temporal, visual ou de fusão.")
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        by_match[row["match_id"]].append(row)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    rows: list[dict[str, object]] = []
    all_goals = sum(len(values) for values in goals_by_match.values())
    for method in methods:
        for limit in sorted(set(args.candidate_limits)):
            covered = 0
            for match_id, goals in goals_by_match.items():
                selected = sorted(by_match[match_id], key=lambda row: float(row[method]), reverse=True)[:limit]
                for goal in goals:
                    covered += int(any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in selected))
            rows.append({"method": method, "k": limit, "catalog_goals": all_goals, "covered_goals": covered, "recall_at_k": f"{covered / all_goals:.6f}", "review_minutes_per_match": f"{limit:.2f}"})
    summary = {"experiment_id": args.output.name, "scope": f"somente {args.expected_split.upper()}; pool temporal top 60 e modelos visual/fusão congelados", "delivery_window": "20+40", "warning": "Mede redução sem treinar pesos; TEST não foi acessado."}
    args.output.mkdir(parents=True)
    with (args.output / "reduction_curve_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
