"""Recalcula o resultado do tracking existente com vários limiares temporais."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analisa limiares temporais do tracking.")
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = read_csv(args.tracking / "frame_tracking.csv")
    annotations = [row for row in read_csv(args.annotations) if row.get("annotation_status") == "completed"]
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frames:
        by_block[row["block_id"]].append(row)
    summaries: list[dict[str, object]] = []
    for threshold in args.thresholds:
        results: list[dict[str, object]] = []
        for annotation in annotations:
            rows = [row for row in by_block[annotation["block_id"]] if float(annotation["start_seconds"]) <= float(row["timestamp_seconds"]) <= float(annotation["end_seconds"])]
            rate = sum(row["track_active"] == "True" for row in rows) / len(rows) if rows else 0.0
            positive = rate >= threshold
            label = annotation["interval_label"]
            if label in {"inconclusive", "not_evaluable"}:
                result = "inconclusive"
            elif label == "goal_structure":
                result = "true_positive" if positive else "false_negative"
            else:
                result = "false_positive" if positive else "true_negative"
            results.append({"threshold": threshold, "interval_id": annotation["interval_id"], "match_id": annotation["match_id"], "capture_domain": annotation["capture_domain"] if "capture_domain" in annotation else "", "interval_label": label, "appearance_rate": round(rate, 6), "result": result})
        evaluable = [row for row in results if row["result"] != "inconclusive"]
        summaries.append({"threshold": threshold, "intervals_total": len(results), "evaluable": len(evaluable), "true_positive": sum(row["result"] == "true_positive" for row in evaluable), "true_negative": sum(row["result"] == "true_negative" for row in evaluable), "false_positive": sum(row["result"] == "false_positive" for row in evaluable), "false_negative": sum(row["result"] == "false_negative" for row in evaluable), "inconclusive": sum(row["result"] == "inconclusive" for row in results)})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tracking_sweep_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"thresholds={len(summaries)}")
    print(f"csv={args.output / 'tracking_sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
