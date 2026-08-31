"""Calcula métricas básicas de continuidade e trajetória do tracking."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def center(row: dict[str, str]) -> tuple[float, float] | None:
    values = [row.get(f"track_{axis}") for axis in ("x1", "y1", "x2", "y2")]
    if any(value in {None, ""} for value in values):
        return None
    return ((float(values[0]) + float(values[2])) / 2, (float(values[1]) + float(values[3])) / 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analisa continuidade e trajetória de um tracking.")
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    args = parser.parse_args()
    frames = read_csv(args.tracking / "frame_tracking.csv")
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frames:
        by_block[row["block_id"]].append(row)
    summaries: list[dict[str, object]] = []
    for block_id, rows in by_block.items():
        active = [row for row in rows if row["track_active"] == "True" and center(row) is not None]
        detected = sum(row["detected"] == "True" for row in rows)
        recovered = sum(row["detected"] == "False" and row["track_active"] == "True" for row in rows)
        streaks: list[int] = []
        current = 0
        for row in rows:
            if row["track_active"] == "True":
                current += 1
            elif current:
                streaks.append(current)
                current = 0
        if current:
            streaks.append(current)
        distances: list[float] = []
        for previous, current_row in zip(active, active[1:]):
            first, second = center(previous), center(current_row)
            if first and second:
                distances.append(math.hypot(second[0] - first[0], second[1] - first[1]))
        total_distance = sum(distances)
        duration = (float(rows[-1]["timestamp_seconds"]) - float(rows[0]["timestamp_seconds"])) if len(rows) > 1 else 0.0
        summaries.append({
            "block_id": block_id,
            "match_id": rows[0]["match_id"],
            "capture_domain": rows[0]["capture_domain"],
            "frames": len(rows),
            "active_frames": len(active),
            "detected_frames": detected,
            "recovered_gap_frames": recovered,
            "active_rate": round(len(active) / len(rows), 6) if rows else 0.0,
            "track_segments": len(streaks),
            "longest_track_frames": max(streaks, default=0),
            "longest_track_seconds": round(max(streaks, default=0) / args.sample_fps, 3),
            "gaps_active": sum(1 for row in rows if row["track_active"] == "True" and row["detected"] == "False"),
            "path_distance_px": round(total_distance, 3),
            "mean_step_px": round(sum(distances) / len(distances), 3) if distances else 0.0,
            "mean_speed_px_s": round((sum(distances) / len(distances)) * args.sample_fps, 3) if distances else 0.0,
            "duration_seconds": round(duration, 3),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trajectory_by_block.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    totals = {
        "blocks": len(summaries),
        "frames": len(frames),
        "active_frames": sum(row["active_frames"] for row in summaries),
        "detected_frames": sum(row["detected_frames"] for row in summaries),
        "recovered_gap_frames": sum(row["recovered_gap_frames"] for row in summaries),
        "track_segments": sum(row["track_segments"] for row in summaries),
        "longest_track_frames": max(row["longest_track_frames"] for row in summaries),
    }
    (args.output / "trajectory_summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False))
    print(f"csv={args.output / 'trajectory_by_block.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
