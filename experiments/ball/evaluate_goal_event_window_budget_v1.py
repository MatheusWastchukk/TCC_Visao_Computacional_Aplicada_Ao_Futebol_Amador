"""Avalia orçamento de revisão por janela e K, exclusivamente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_window(value: str) -> tuple[float, float]:
    try:
        before, after = (float(part) for part in value.split(":", 1))
    except ValueError as error:
        raise ValueError(f"Janela inválida: {value}; use antes:depois.") from error
    if before < 0 or after <= 0:
        raise ValueError("A janela exige antes >= 0 e depois > 0.")
    return before, after


def select_diverse(rows: list[dict[str, str]], separation: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if all(abs(timestamp - float(other["best_timestamp_seconds"])) >= separation for other in selected):
            selected.append(row)
    return selected


def covers(row: dict[str, str], goal: float, before: float, after: float) -> bool:
    center = float(row["best_timestamp_seconds"])
    return center - before <= goal <= center + after


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    output: list[list[float]] = []
    for start, end in sorted(intervals):
        if not output or start > output[-1][1]:
            output.append([start, end])
        else:
            output[-1][1] = max(output[-1][1], end)
    return [(start, end) for start, end in output]


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir vídeo TRAIN: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Metadados inválidos: {path}")
    return frames / fps


def pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        dominated = any(
            other is not row
            and int(other["covered_goals"]) >= int(row["covered_goals"])
            and float(other["unique_review_seconds"]) <= float(row["unique_review_seconds"])
            and (int(other["covered_goals"]) > int(row["covered_goals"]) or float(other["unique_review_seconds"]) < float(row["unique_review_seconds"]))
            for other in rows
        )
        if not dominated:
            output.append(row)
    return sorted(output, key=lambda item: (float(item["unique_review_seconds"]), -int(item["covered_goals"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia Recall × minutos únicos para janelas de candidatos TRAIN.")
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 999999])
    parser.add_argument("--windows", nargs="+", default=[
        "0:20", "5:15", "10:10", "0:30", "5:25", "10:20", "15:15", "20:10",
        "0:40", "5:35", "10:30", "15:25", "20:20", "0:45", "10:35", "15:30",
        "20:25", "0:60", "10:50", "20:40", "30:30",
    ])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    predictions = read_csv(args.rank_predictions)
    if "model" in predictions[0]:
        predictions = [row for row in predictions if row["model"] == "random_forest"]
    catalog = read_csv(args.goal_timestamps)
    if any(row.get("split") != "train" for row in catalog):
        raise AssertionError("O experimento aceita somente timestamps TRAIN.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    videos_by_match: dict[str, str] = {}
    for row in catalog:
        videos_by_match[row["match_id"]] = row["video_filename"]
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    ranked_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        ranked_by_match[row["match_id"]].append(row)
    if set(goals_by_match) != set(ranked_by_match):
        raise ValueError("Catálogo e predições não possuem as mesmas partidas TRAIN.")
    diverse_by_match = {match: select_diverse(rows, args.min_separation_seconds) for match, rows in ranked_by_match.items()}
    durations = {match: video_duration(args.videos_dir / videos_by_match[match]) for match in ranked_by_match}
    windows = sorted(set(parse_window(value) for value in args.windows), key=lambda item: (sum(item), item[0]))
    limits = sorted(set(args.candidate_limits))
    catalog_goals = sum(len(values) for values in goals_by_match.values())
    rows: list[dict[str, object]] = []
    for before, after in windows:
        full_pool_covered = sum(
            any(covers(row, goal, before, after) for row in ranked_by_match[match])
            for match, goals in goals_by_match.items() for goal in goals
        )
        diverse_pool_covered = sum(
            any(covers(row, goal, before, after) for row in diverse_by_match[match])
            for match, goals in goals_by_match.items() for goal in goals
        )
        for limit in limits:
            chosen_by_match = {match: candidates[:limit] for match, candidates in diverse_by_match.items()}
            covered = sum(
                any(covers(row, goal, before, after) for row in chosen_by_match[match])
                for match, goals in goals_by_match.items() for goal in goals
            )
            merged_by_match = {
                match: merge([(max(0.0, float(row["best_timestamp_seconds"]) - before), min(durations[match], float(row["best_timestamp_seconds"]) + after)) for row in candidates])
                for match, candidates in chosen_by_match.items()
            }
            unique_seconds = sum(end - start for intervals in merged_by_match.values() for start, end in intervals)
            raw_seconds = sum((min(durations[match], float(row["best_timestamp_seconds"]) + after) - max(0.0, float(row["best_timestamp_seconds"]) - before)) for match, candidates in chosen_by_match.items() for row in candidates)
            total_video_seconds = sum(durations.values())
            rows.append({
                "window_before_seconds": before, "window_after_seconds": after, "window_total_seconds": before + after,
                "k_limit": "pool_completo" if limit >= max(len(value) for value in diverse_by_match.values()) else limit,
                "catalog_goals": catalog_goals, "full_candidate_pool_goals": full_pool_covered,
                "diverse_pool_goals": diverse_pool_covered, "covered_goals": covered,
                "recall": f"{covered / catalog_goals:.6f}", "candidates": sum(len(value) for value in chosen_by_match.values()),
                "merged_segments": sum(len(value) for value in merged_by_match.values()),
                "raw_review_seconds": f"{raw_seconds:.3f}", "unique_review_seconds": f"{unique_seconds:.3f}",
                "unique_review_minutes_per_match": f"{unique_seconds / 60 / len(chosen_by_match):.3f}",
                "unique_video_reduction_percent": f"{100.0 * (1.0 - unique_seconds / total_video_seconds):.3f}",
            })
    pareto_rows = pareto(rows)
    args.output.mkdir(parents=True)
    write_csv(args.output / "window_budget_curve_v1.csv", rows)
    write_csv(args.output / "pareto_frontier_v1.csv", pareto_rows)
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN; sem retreinamento",
        "ranking": "predições Random Forest leave-one-match-out; diversidade temporal de 45 s",
        "reconciliation": {
            "full_candidate_pool": "todos os grupos temporais pontuados; mede o teto de recuperação",
            "diverse_ranked_pool": "lista temporalmente diversa e ordenada; é a população entregue ao revisor",
        },
        "catalog_goals": catalog_goals, "windows": [f"{before:g}+{after:g}" for before, after in windows],
        "candidate_limits": limits, "pareto_points": len(pareto_rows),
        "warning": "Resultados de desenvolvimento em TRAIN. Não altera V4 nem usa VALIDATION ou TEST.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
