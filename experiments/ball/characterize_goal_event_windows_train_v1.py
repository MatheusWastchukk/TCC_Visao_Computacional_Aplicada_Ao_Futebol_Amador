"""Caracteriza janelas de entrega sem alterar o pool ou o ranking V4."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def merged_seconds(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def main() -> int:
    parser = argparse.ArgumentParser(description="Caracteriza janelas de candidatos em TRAIN sem recalibrar ranking.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", nargs="+", default=["5:10", "10:20", "15:30", "20:40"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    candidates = read_csv(args.candidates)
    if not candidates or any(row.get("split") != "train" for row in candidates):
        raise AssertionError("A caracterização aceita somente candidatos TRAIN.")
    goals: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(args.goal_timestamps):
        if row.get("split") == "train" and row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds"):
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    windows: list[tuple[float, float]] = []
    for value in args.windows:
        try:
            before, after = (float(part) for part in value.split(":", 1))
        except ValueError as error:
            raise ValueError(f"Janela inválida: {value}; use antes:depois.") from error
        if before < 0 or after <= 0:
            raise ValueError("As janelas devem ter antes >= 0 e depois > 0.")
        windows.append((before, after))
    rows: list[dict[str, object]] = []
    for before, after in windows:
        covered = 0
        intervals_by_match: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for candidate in candidates:
            timestamp = float(candidate["event_timestamp_seconds"])
            intervals_by_match[candidate["match_id"]].append((max(0.0, timestamp - before), timestamp + after))
        for match_id, timestamps in goals.items():
            intervals = intervals_by_match[match_id]
            covered += sum(any(start <= goal <= end for start, end in intervals) for goal in timestamps)
        seconds = sum(merged_seconds(intervals) for intervals in intervals_by_match.values())
        rows.append({
            "window_before_seconds": before, "window_after_seconds": after,
            "catalog_goals": sum(len(values) for values in goals.values()), "covered_goals": covered,
            "coverage": f"{covered / sum(len(values) for values in goals.values()):.6f}",
            "unique_review_seconds": f"{seconds:.3f}",
            "unique_review_minutes_per_match": f"{seconds / 60 / len(intervals_by_match):.3f}",
        })
    args.output.mkdir(parents=True)
    with (args.output / "window_characterization_train_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"experiment_id": args.output.name, "scope": "somente TRAIN; caracterização sem alterar V4", "windows": rows}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
