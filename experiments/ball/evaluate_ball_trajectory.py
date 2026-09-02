"""Mede cobertura, identidade, erro espacial e continuidade de trajetórias anotadas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def normalized_distance(first: tuple[float, float, float, float], second: tuple[float, float, float, float], diagonal: float) -> float:
    first_center, second_center = center(first), center(second)
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]) / diagonal


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia a continuidade espacial de trajetórias da bola.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-matches", nargs="+", default=None)
    args = parser.parse_args()

    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequence_rows = [
            row for row in csv.DictReader(file)
            if row.get("ball_visibility", "VISIBLE") in {"VISIBLE", "PARTIAL"} and row.get("ball_of_play_box_json")
        ]
    with args.results.open(newline="", encoding="utf-8") as file:
        result_rows = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    if args.include_matches:
        matches = set(args.include_matches)
        sequence_rows = [row for row in sequence_rows if row["match_id"] in matches]
        if not sequence_rows:
            parser.error("Nenhuma anotação corresponde às partidas informadas.")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sequence_rows:
        grouped[row["sequence_id"]].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    frame_metrics: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["sequence_frame"]))
        correct_values: list[bool] = []
        candidate_values: list[bool] = []
        errors: list[float] = []
        for row in rows:
            result = result_rows[(sequence_id, row["sequence_frame"])]
            target = tuple(float(value) for value in json.loads(row["ball_of_play_box_json"]))
            selected = tuple(float(value) for value in json.loads(result["selected_box_json"])) if result.get("selected_box_json") else None
            image = cv2.imread(row["image_path"])
            if image is None:
                raise FileNotFoundError(row["image_path"])
            diagonal = math.hypot(image.shape[0], image.shape[1])
            candidate_present = result["target_candidate_rank"] != "0"
            correct = result["identity_correct"] == "True"
            error = normalized_distance(selected, target, diagonal) if selected else None
            candidate_values.append(candidate_present)
            correct_values.append(correct)
            if error is not None:
                errors.append(error)
            frame_metrics.append({
                "sequence_id": sequence_id, "sequence_frame": row["sequence_frame"], "sample_id": row["sample_id"],
                "target_candidate_rank": result["target_candidate_rank"], "selected_candidate_rank": result["selected_candidate_rank"],
                "identity_correct": str(correct), "center_error_normalized": f"{error:.6f}" if error is not None else "",
            })
        summary_rows.append({
            "sequence_id": sequence_id, "frames": str(len(rows)),
            "candidate_recall": f"{sum(candidate_values) / len(rows):.6f}",
            "identity_accuracy": f"{sum(correct_values) / len(rows):.6f}",
            "longest_correct_run_frames": str(longest_run(correct_values)),
            "longest_correct_run_seconds": f"{longest_run(correct_values) / 5.0:.3f}",
            "mean_center_error_normalized": f"{sum(errors) / len(errors):.6f}" if errors else "",
            "max_center_error_normalized": f"{max(errors):.6f}" if errors else "",
        })

    for filename, rows in (("trajectory_frame_metrics.csv", frame_metrics), ("trajectory_summary.csv", summary_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    total_frames = len(frame_metrics)
    metrics = {
        "experiment_id": args.output.name,
        "sequences": len(summary_rows),
        "target_frames": total_frames,
        "candidate_recall": sum(row["target_candidate_rank"] != "0" for row in frame_metrics) / total_frames,
        "identity_accuracy": sum(row["identity_correct"] == "True" for row in frame_metrics) / total_frames,
        "mean_center_error_normalized": sum(float(row["center_error_normalized"]) for row in frame_metrics if row["center_error_normalized"]) / total_frames,
    }
    (args.output / "trajectory_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
