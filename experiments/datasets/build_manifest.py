"""Cria o manifesto imutável de frames usado pelos experimentos geométricos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.timestamps.open(newline="", encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "frame_id",
            "match_id",
            "timestamp",
            "original_label",
            "goal_structure_label",
            "split",
            "file_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            timestamp = float(row["timestamp_seconds"])
            label = row["label"]
            frame_id = f"{row['match_id']}_{timestamp:010.3f}"
            split = {
                "match_01": "development",
                "match_02": "validation",
                "match_03": "test",
            }[row["match_id"]]
            frame_file = f"{timestamp:010.3f}"
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "match_id": row["match_id"],
                    "timestamp": f"{timestamp:.3f}",
                    "original_label": label,
                    "goal_structure_label": (
                        "goal_structure" if label in {"main_goal", "secondary_goal"} else "not_goal"
                    ),
                    "split": split,
                    "file_path": (
                        f"experiments/goal_detection_pilot/data/frames/{label}/"
                        f"{row['match_id']}_{frame_file}.jpg"
                    ),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
