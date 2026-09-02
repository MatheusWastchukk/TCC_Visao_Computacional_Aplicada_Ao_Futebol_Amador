"""Resume a tendência temporal da distância entre bola rastreada e goleira detectada."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def classify(slope: float, minimum_change: float) -> str:
    """Classifica apenas a tendência espacial; não infere evento de gol."""
    if slope <= -minimum_change:
        return "APROXIMACAO"
    if slope >= minimum_change:
        return "AFASTAMENTO"
    return "ESTAVEL_OU_AMBIGUA"


def main() -> int:
    parser = argparse.ArgumentParser(description="Mede aproximação espacial da bola em direção à goleira por clipe.")
    parser.add_argument("--relation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-slope-pixels-frame", type=float, default=2.0)
    args = parser.parse_args()
    with args.relation.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_id"]].append(row)
    summaries: list[dict[str, str]] = []
    for sequence_id, frames in sorted(grouped.items()):
        frames.sort(key=lambda row: int(row["sequence_frame"]))
        valid = [(int(row["sequence_frame"]), float(row["ball_goal_center_distance"])) for row in frames if row["ball_goal_center_distance"]]
        if len(valid) < 2:
            trend, slope, first, last, descending = "INSUFICIENTE", None, None, None, None
        else:
            first, last = valid[0][1], valid[-1][1]
            slope = (last - first) / (valid[-1][0] - valid[0][0])
            steps = [current[1] - previous[1] for previous, current in zip(valid, valid[1:]) if current[0] == previous[0] + 1]
            descending = sum(step < 0 for step in steps) / len(steps) if steps else 0.0
            trend = classify(slope, args.minimum_slope_pixels_frame)
        summaries.append({
            "sequence_id": sequence_id, "match_id": frames[0]["match_id"], "frames": str(len(frames)),
            "frames_with_goal": str(len(valid)), "first_distance_pixels": f"{first:.3f}" if first is not None else "",
            "last_distance_pixels": f"{last:.3f}" if last is not None else "", "slope_pixels_per_frame": f"{slope:.3f}" if slope is not None else "",
            "descending_adjacent_fraction": f"{descending:.3f}" if descending is not None else "", "spatial_trend": trend,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "approach_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    metrics = {
        "experiment_id": args.output.name, "sequences": len(summaries),
        "trend_counts": {trend: sum(row["spatial_trend"] == trend for row in summaries) for trend in sorted({row["spatial_trend"] for row in summaries})},
        "interpretation": "Tendência espacial exploratória; não representa decisão de gol.",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
