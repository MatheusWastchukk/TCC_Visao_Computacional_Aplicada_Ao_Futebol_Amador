"""Aplica o ranking temporal congelado e avalia cobertura em um split fechado."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def select_diverse(rows: list[dict[str, object]], limit: int, separation: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Aplica o ranker temporal final sem retreinamento.")
    parser.add_argument("--temporal-dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["validation"], required=True)
    parser.add_argument("--max-candidates-per-match", type=int, default=4)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.max_candidates_per_match < 1 or args.min_separation_seconds < 0:
        raise ValueError("Os limites de seleção devem ser válidos.")

    groups, fields = read_csv(args.temporal_dataset)
    required = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    if missing := required - set(fields):
        raise ValueError(f"Dataset temporal sem colunas: {sorted(missing)}")
    frozen = joblib.load(args.model)
    feature_names = frozen["features"]
    if missing := set(feature_names) - set(fields):
        raise ValueError(f"Dataset incompatível com o ranker congelado: {sorted(missing)}")
    features = np.array([[float(row[name]) for name in feature_names] for row in groups], dtype=np.float32)
    scores = frozen["model"].predict_proba(features)[:, 1]

    split_rows, split_fields = read_csv(args.split_manifest)
    if {"match_id", "video_filename", "split"} - set(split_fields):
        raise ValueError("Manifesto de split incompleto.")
    videos = {
        row["match_id"]: row["video_filename"]
        for row in split_rows if row["split"] == args.expected_split
    }
    group_matches = {row["match_id"] for row in groups}
    if not group_matches or group_matches - set(videos):
        raise AssertionError("Há grupos fora do split VALIDATION esperado.")

    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"goal_catalog_id", "match_id", "split", "event_timestamp_seconds", "annotation_status"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Catálogo sem colunas: {sorted(missing)}")
    if any(row["split"] != args.expected_split for row in catalog):
        raise AssertionError("O catálogo contém split diferente de VALIDATION.")
    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(row)

    ranked = [{**row, "rank_score": f"{float(score):.8f}"} for row, score in zip(groups, scores)]
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ranked:
        by_match[str(row["match_id"])].append(row)
    selected = [row for match_id in sorted(by_match) for row in select_diverse(by_match[match_id], args.max_candidates_per_match, args.min_separation_seconds)]

    events: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        timestamp = float(row["best_timestamp_seconds"])
        events.append({
            "event_id": f"validation_ranker_candidate_v1_{index:04d}",
            "source_file": "score_goal_event_pipeline_v1.py",
            "source_interval_id": row["group_uid"],
            "source_interval_label": "ranker_temporal_congelado",
            "match_id": row["match_id"], "video_filename": videos[str(row["match_id"])],
            "split": args.expected_split, "event_timestamp_seconds": f"{timestamp:.3f}",
            "event_time_hhmmss": format_time(timestamp), "candidate_start_seconds": f"{max(0.0, timestamp - args.clip_pre_seconds):.3f}",
            "candidate_end_seconds": f"{timestamp + args.clip_post_seconds:.3f}", "rank_score": row["rank_score"],
            "annotation_status": "pending", "ground_truth": "",
            "notes": "Candidato do ranker temporal congelado; revisão humana ainda é independente.",
        })

    associations: list[dict[str, object]] = []
    covered = 0
    for match_id, goal_rows in sorted(goals_by_match.items()):
        candidates = [event for event in events if event["match_id"] == match_id]
        for goal in goal_rows:
            timestamp = float(goal["event_timestamp_seconds"])
            matching = [event["event_id"] for event in candidates if float(event["candidate_start_seconds"]) <= timestamp <= float(event["candidate_end_seconds"])]
            covered += int(bool(matching))
            associations.append({"goal_catalog_id": goal["goal_catalog_id"], "match_id": match_id, "goal_timestamp_seconds": f"{timestamp:.3f}", "covered_by_candidate": str(bool(matching)), "candidate_event_ids": ";".join(matching)})

    args.output.mkdir(parents=True)
    write_csv(args.output / "ranked_groups_v1.csv", ranked, [*fields, "rank_score"])
    event_fields = list(events[0]) if events else ["event_id"]
    write_csv(args.output / "candidate_events_v1.csv", events, event_fields)
    write_csv(args.output / "candidate_goal_associations_v1.csv", associations, ["goal_catalog_id", "match_id", "goal_timestamp_seconds", "covered_by_candidate", "candidate_event_ids"])
    summary = {
        "experiment_id": args.output.name, "scope": "VALIDATION; configuração TRAIN congelada", "ranker_model": "RandomForest 300/min_leaf=2/balanced",
        "groups_scored": len(ranked), "selected_candidates": len(events), "selected_by_match": {match: len([event for event in events if event["match_id"] == match]) for match in sorted(videos)},
        "catalog_goals_evaluable": len(associations), "catalog_goals_covered": covered,
        "recall_at_k": covered / len(associations) if associations else None,
        "warning": "Resultado de VALIDATION com configuração congelada; não use para retreinar, ajustar limiares ou selecionar novo modelo.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
