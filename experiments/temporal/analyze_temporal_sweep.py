"""Analisa resultados temporais existentes por limiar, partida e domínio."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def classify(label: str, confirmed: bool) -> str:
    if label in {"inconclusive", "not_evaluable"}:
        return "inconclusive"
    if label == "goal_structure":
        return "true_positive" if confirmed else "false_negative"
    return "false_positive" if confirmed else "true_negative"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analisa um sweep temporal já processado.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = [row for row in read_csv(args.annotations) if row.get("annotation_status") == "completed"]
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for run in args.runs:
        frames = read_csv(run / "frame_predictions.csv")
        by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in frames:
            by_block[row["block_id"]].append(row)
        for threshold in args.thresholds:
            interval_rows: list[dict[str, object]] = []
            for annotation in annotations:
                rows = [row for row in by_block[annotation["block_id"]] if float(annotation["start_seconds"]) - 1e-6 <= float(row["timestamp_seconds"]) <= float(annotation["end_seconds"]) + 1e-6]
                rate = sum(row["detected"] == "True" for row in rows) / len(rows) if rows else 0.0
                result = classify(annotation["interval_label"], rate >= threshold)
                interval_rows.append({
                    "model_run": run.name,
                    "threshold": threshold,
                    "interval_id": annotation["interval_id"],
                    "match_id": annotation["match_id"],
                    "capture_domain": next(row["capture_domain"] for row in frames if row["block_id"] == annotation["block_id"]),
                    "interval_label": annotation["interval_label"],
                    "frames_analyzed": len(rows),
                    "appearance_rate": round(rate, 6),
                    "result": result,
                })
            all_rows.extend(interval_rows)
            evaluable = [row for row in interval_rows if row["result"] != "inconclusive"]
            summaries.append({
                "model_run": run.name,
                "threshold": threshold,
                "intervals_total": len(interval_rows),
                "intervals_evaluable": len(evaluable),
                "true_positive": sum(row["result"] == "true_positive" for row in evaluable),
                "true_negative": sum(row["result"] == "true_negative" for row in evaluable),
                "false_positive": sum(row["result"] == "false_positive" for row in evaluable),
                "false_negative": sum(row["result"] == "false_negative" for row in evaluable),
                "inconclusive": sum(row["result"] == "inconclusive" for row in interval_rows),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "interval_sweep.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    print(f"intervals={len(all_rows)}")
    print(f"summary={args.output / 'sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
