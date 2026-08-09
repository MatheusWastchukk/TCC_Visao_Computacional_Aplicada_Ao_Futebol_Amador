"""Extract pilot frames from timestamp manifests.

Expected CSV columns:
match_id,video_path,timestamp_seconds,label
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames for the goal detection pilot.")
    parser.add_argument("--timestamps", required=True, type=Path, help="CSV with frame timestamps.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where frames will be saved.",
    )
    return parser.parse_args()


def format_timestamp(timestamp_seconds: float) -> str:
    milliseconds = int(round(timestamp_seconds * 1000))
    seconds, ms = divmod(milliseconds, 1000)
    return f"{seconds:06d}.{ms:03d}"


def extract_frame(video_path: Path, timestamp_seconds: float, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame at {timestamp_seconds:.3f}s from {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise ValueError(f"Could not write frame: {output_path}")
    finally:
        capture.release()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir

    with args.timestamps.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"match_id", "video_path", "timestamp_seconds", "label"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            match_id = row["match_id"].strip()
            label = row["label"].strip() or "unlabeled"
            timestamp_seconds = float(row["timestamp_seconds"])
            video_path = Path(row["video_path"])
            timestamp_label = format_timestamp(timestamp_seconds)
            output_path = output_dir / label / f"{match_id}_{timestamp_label}.jpg"
            extract_frame(video_path, timestamp_seconds, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
