"""Prepare event-window frames as input to the existing multi-hypothesis tracker."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXP = Path(__file__).resolve().parent
DEFAULT_INPUT = EXP / "oracle_frame_annotations_v1_annotated.csv"
DEFAULT_OUTPUT = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "oracle_tracker_input_frames_16_26.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    selected = [row for row in rows if 16 <= int(row["sequence_frame"]) <= 26]
    selected.sort(key=lambda row: (row["sequence_id"], int(row["sequence_frame"])))
    if len(rows) != 1640 or len(selected) != 440 or len({row["sequence_id"] for row in selected}) != 40:
        raise ValueError(f"Unexpected frozen oracle input: all={len(rows)}, selected={len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Prepared {len(selected)} frames in {len({row['sequence_id'] for row in selected})} sequences: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
