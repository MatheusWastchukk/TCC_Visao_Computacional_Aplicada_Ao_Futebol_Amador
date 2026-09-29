"""Avalia XGBoost rank:pairwise com utilidade do clipe como alvo, só em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from xgboost import XGBRanker


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para salvar: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utility_grade(center: float, goals: list[float], before: float, after: float) -> int:
    grades = []
    for goal in goals:
        position = (goal - (center - before)) / (before + after)
        if 0 <= position <= 1:
            grades.append(1 if position <= 0.10 or position >= 0.90 else 2 if position <= 0.25 or position >= 0.75 else 3)
    return max(grades, default=0)


def select_diverse(rows: list[dict[str, object]], limit: int, gap: float) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        center = float(row["best_timestamp_seconds"])
        if any(abs(center - float(other["best_timestamp_seconds"])) < gap for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia XGBoost LTR com alvo graded de utilidade do clipe em LOMO TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[30, 40, 50, 60, 70, 80, 100])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=20.0)
    parser.add_argument("--clip-post-seconds", type=float, default=40.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    rows, fields = read_csv(args.dataset)
    required = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    if missing := required - set(fields):
        raise ValueError(f"Dataset sem colunas necessárias: {sorted(missing)}")
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    if "split" not in catalog_fields or any(row["split"].lower() != "train" for row in catalog):
        raise AssertionError("O experimento aceita exclusivamente o catálogo TRAIN.")
    goals: dict[str, list[float]] = defaultdict(list)
    filenames: dict[str, str] = {}
    for row in catalog:
        filenames[row["match_id"]] = row["video_filename"]
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    matches = sorted({row["match_id"] for row in rows})
    if set(matches) != set(goals):
        raise ValueError("Dataset temporal e catálogo não possuem as mesmas partidas TRAIN.")
    features = [field for field in fields if field not in required]
    x = np.asarray([[float(row[field]) for field in features] for row in rows], dtype=np.float32)
    y = np.asarray([utility_grade(float(row["best_timestamp_seconds"]), goals[row["match_id"]], args.clip_pre_seconds, args.clip_post_seconds) for row in rows], dtype=np.int32)
    match_array = np.asarray([row["match_id"] for row in rows])
    scores = np.zeros(len(rows), dtype=np.float64)
    configuration = {
        "objective": "rank:pairwise", "n_estimators": 250, "max_depth": 4,
        "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 1.0, "reg_lambda": 1.0, "tree_method": "hist",
        "device": "cpu", "random_state": 42, "n_jobs": 4,
    }
    for held_match in matches:
        test_mask = match_array == held_match
        train_indices = np.where(~test_mask)[0]
        order = train_indices[np.argsort(match_array[train_indices], kind="stable")]
        ordered_matches = match_array[order]
        group_sizes = [int(np.sum(ordered_matches == match)) for match in sorted(set(ordered_matches))]
        model = XGBRanker(**configuration)
        model.fit(x[order], y[order], group=group_sizes, verbose=False)
        scores[test_mask] = model.predict(x[test_mask])

    prediction_rows = [{**row, "clip_utility_grade": int(grade), "rank_score": f"{score:.8f}"}
                       for row, grade, score in zip(rows, y, scores)]
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        by_match[str(row["match_id"])].append(row)
    selected = {match: select_diverse(values, max(args.candidate_limits), args.min_separation_seconds)
                for match, values in by_match.items()}

    durations: dict[str, float] = {}
    for match, filename in filenames.items():
        video = cv2.VideoCapture(str(args.videos_dir / filename))
        if not video.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir vídeo TRAIN: {filename}")
        fps, frames = float(video.get(cv2.CAP_PROP_FPS)), float(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        if fps <= 0 or frames <= 0:
            raise ValueError(f"Metadados inválidos no vídeo: {filename}")
        durations[match] = frames / fps

    goal_ranks = []
    for match, timestamps in goals.items():
        for timestamp in timestamps:
            ranks = [index for index, row in enumerate(selected[match], 1)
                     if float(row["best_timestamp_seconds"]) - args.clip_pre_seconds <= timestamp <= float(row["best_timestamp_seconds"]) + args.clip_post_seconds]
            goal_ranks.append({"match_id": match, "goal_timestamp_seconds": f"{timestamp:.3f}",
                               "best_diverse_candidate_rank": ranks[0] if ranks else ""})
    curve = []
    for limit in sorted(set(args.candidate_limits)):
        chosen = {match: values[:limit] for match, values in selected.items()}
        covered = sum(bool(row["best_diverse_candidate_rank"]) and int(row["best_diverse_candidate_rank"]) <= limit for row in goal_ranks)
        intervals = {match: merge([(max(0.0, float(row["best_timestamp_seconds"]) - args.clip_pre_seconds),
                                   min(durations[match], float(row["best_timestamp_seconds"]) + args.clip_post_seconds)) for row in values])
                     for match, values in chosen.items()}
        unique_seconds = sum(end - start for values in intervals.values() for start, end in values)
        curve.append({"k": limit, "catalog_goals": len(goal_ranks), "covered_goals": covered,
                      "recall_at_k": f"{covered / len(goal_ranks):.6f}",
                      "candidates": sum(map(len, chosen.values())),
                      "merged_segments": sum(map(len, intervals.values())),
                      "unique_review_minutes_per_match": f"{unique_seconds / 60 / len(chosen):.3f}"})

    args.output.mkdir(parents=True)
    write_csv(args.output / "lomo_predictions_v1.csv", prediction_rows)
    write_csv(args.output / "goal_ranks_lomo_v1.csv", goal_ranks)
    write_csv(args.output / "recall_review_curve_lomo_v1.csv", curve)
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN; leave-one-match-out; TEST não acessado",
        "objective": "ranking por partida com alvo graded CLIP_USEFUL",
        "target_grades": {"0": "clipe sem gol catalogado", "1": "gol na borda do clipe", "2": "gol com contexto limitado", "3": "gol com bom contexto"},
        "target_counts": {str(value): int(np.sum(y == value)) for value in sorted(set(y))},
        "window_seconds": {"before": args.clip_pre_seconds, "after": args.clip_post_seconds},
        "minimum_diversity_gap_seconds": args.min_separation_seconds,
        "features": features, "configuration": configuration,
        "warning": "Desenvolvimento TRAIN com previsões LOMO; não é métrica de generalização nem confirmação oficial de gol.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(curve, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
