"""Seleciona negativos difíceis por predições LOMO, exclusivamente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara negativos difíceis a partir do ranking LOMO TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-top-k", type=int, default=20)
    parser.add_argument("--common-negative-multiplier", type=int, default=5)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.hard_top_k < 1 or args.common_negative_multiplier < 0:
        raise ValueError("Os limites de seleção devem ser não negativos e válidos.")

    groups, fields = read_csv(args.dataset)
    required_groups = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    if missing := required_groups - set(fields):
        raise ValueError(f"Dataset sem colunas: {sorted(missing)}")
    predictions, prediction_fields = read_csv(args.rank_predictions)
    required_predictions = {"model", "group_uid", "match_id", "best_timestamp_seconds", "rank_score"}
    if missing := required_predictions - set(prediction_fields):
        raise ValueError(f"Predições sem colunas: {sorted(missing)}")
    predictions = [row for row in predictions if row["model"] == "random_forest"]
    score_by_group = {row["group_uid"]: row for row in predictions}
    if {row["group_uid"] for row in groups} != set(score_by_group):
        raise AssertionError("As predições LOMO não correspondem exatamente aos grupos TRAIN.")
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"match_id", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Catálogo sem colunas: {sorted(missing)}")
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("A seleção de hard negatives aceita somente TRAIN.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    predictions_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        predictions_by_match[row["match_id"]].append(row)
    hard_ids = {
        row["group_uid"]
        for match_rows in predictions_by_match.values()
        for row in select_diverse(match_rows, args.hard_top_k, args.min_separation_seconds)
    }
    enriched: list[dict[str, object]] = []
    for group in groups:
        timestamp = float(group["best_timestamp_seconds"])
        contains_goal = any(timestamp - args.clip_pre_seconds <= goal <= timestamp + args.clip_post_seconds for goal in goals_by_match[group["match_id"]])
        weak_label = int(group["weak_label"])
        if weak_label:
            role = "positive"
        elif contains_goal:
            role = "ambiguous_negative_excluded"
        elif group["group_uid"] in hard_ids:
            role = "hard_negative"
        else:
            role = "common_negative_candidate"
        enriched.append({**group, "base_lomo_rank_score": score_by_group[group["group_uid"]]["rank_score"], "contains_catalog_goal_in_clip": str(contains_goal), "selection_role": role})

    positives = [row for row in enriched if row["selection_role"] == "positive"]
    hard_negatives = [row for row in enriched if row["selection_role"] == "hard_negative"]
    common_negatives = [row for row in enriched if row["selection_role"] == "common_negative_candidate"]
    target_common = min(len(common_negatives), args.common_negative_multiplier * len(positives))
    rng = np.random.default_rng(args.random_seed)
    common_indices = rng.choice(len(common_negatives), size=target_common, replace=False) if target_common else []
    selected_ids = {row["group_uid"] for row in positives + hard_negatives}
    selected_ids.update(common_negatives[int(index)]["group_uid"] for index in common_indices)
    for row in enriched:
        row["selected_for_hard_ranker"] = str(row["group_uid"] in selected_ids)

    args.output.mkdir(parents=True)
    output_fields = [*fields, "base_lomo_rank_score", "contains_catalog_goal_in_clip", "selection_role", "selected_for_hard_ranker"]
    write_csv(args.output / "hard_ranker_selection_v1.csv", enriched, output_fields)
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN; seleção baseada em predições LOMO",
        "hard_top_k": args.hard_top_k, "common_negative_multiplier": args.common_negative_multiplier,
        "random_seed": args.random_seed, "roles": dict(Counter(row["selection_role"] for row in enriched)),
        "selected": {"positives": len(positives), "hard_negatives": len(hard_negatives), "common_negatives": target_common, "total": len(selected_ids)},
        "warning": "Negativos ambíguos que contêm timestamp catalogado no clipe são excluídos; VALIDATION e TEST não foram lidos.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
