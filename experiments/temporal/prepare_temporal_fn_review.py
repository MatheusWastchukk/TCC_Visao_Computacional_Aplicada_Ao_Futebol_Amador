"""Prepara frames representativos dos falsos negativos temporais para revisão."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara a revisão visual dos falsos negativos temporais.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    args = parse_args()
    results = read_csv(args.results)
    frame_rows = read_csv(args.frames)
    false_negatives = [row for row in results if row.get("result") == "false_negative"]
    args.output.mkdir(parents=True, exist_ok=True)
    review_rows: list[dict[str, str]] = []

    for result in false_negatives:
        start = float(result["start_seconds"])
        end = float(result["end_seconds"])
        candidates = [
            row
            for row in frame_rows
            if row["block_id"] == result["block_id"]
            and start - 1e-6 <= float(row["timestamp_seconds"]) <= end + 1e-6
        ]
        if not candidates:
            candidates = [
                row
                for row in frame_rows
                if row["block_id"] == result["block_id"]
                and start - 1e-6 <= float(row["timestamp_seconds"]) <= end + 1e-6
            ]
        selected: list[dict[str, str]] = []
        for candidate in [candidates[0], min(candidates, key=lambda row: float(row["confidence"])), candidates[-1]]:
            if candidate not in selected:
                selected.append(candidate)

        video_path = args.videos_dir / result["video_filename"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        for position, candidate in enumerate(selected, start=1):
            timestamp = float(candidate["timestamp_seconds"])
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, image = cap.read()
            if not ok or image is None:
                continue
            filename = f"{result['interval_id']}_{position:02d}_{timestamp:.3f}.jpg"
            output_path = args.output / filename
            cv2.imwrite(str(output_path), image)
            review_rows.append(
                {
                    "interval_id": result["interval_id"],
                    "block_id": result["block_id"],
                    "match_id": result["match_id"],
                    "video_filename": result["video_filename"],
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "timestamp_hhmmss": format_time(timestamp),
                    "detected": candidate["detected"],
                    "confidence": candidate["confidence"],
                    "frame_path": str(output_path),
                    "review_status": "pending",
                    "notes": "",
                }
            )
        cap.release()

    csv_path = args.output / "false_negative_review_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"falsos_negativos={len(false_negatives)}")
    print(f"frames_preparados={len(review_rows)}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
