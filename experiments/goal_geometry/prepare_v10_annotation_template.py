"""Create a spaced timestamp template for new V10 annotations."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a V10 annotation timestamp template.")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--existing-timestamps", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-match", type=int, default=20)
    parser.add_argument(
        "--max-selected-per-match",
        type=int,
        help="Limita a quantidade de candidatos mantidos por partida.",
    )
    return parser.parse_args()


def existing_seconds(paths: list[Path]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                match_id = row["match_id"]
                raw = row.get("timestamp_seconds") or row.get("timestamp") or ""
                if raw:
                    values.setdefault(match_id, []).append(float(raw))
                else:
                    match = re.search(r"(match_\d+)_(\d+\.\d+)", row.get("frame_id", ""))
                    if match:
                        values.setdefault(match.group(1), []).append(float(match.group(2)))
    return values


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Não foi possível obter duração de {path}")
    return frames / fps


def main() -> int:
    args = parse_args()
    previous = existing_seconds(args.existing_timestamps)
    output_rows: list[dict[str, str]] = []
    for video_path in sorted(args.videos_dir.glob("Match_*_A.mp4")):
        match_number = video_path.stem.split("_")[1]
        match_id = f"match_{match_number}"
        duration = video_duration(video_path)
        candidates = [
            duration * index / (args.frames_per_match + 1)
            for index in range(1, args.frames_per_match + 1)
        ]
        selected: list[float] = []
        for timestamp in candidates:
            if all(abs(timestamp - old) >= 10.0 for old in previous.get(match_id, [])):
                selected.append(timestamp)
                if (
                    args.max_selected_per_match
                    and len(selected) >= args.max_selected_per_match
                ):
                    break
        for index, timestamp in enumerate(selected, start=1):
            frame_id = f"{match_id}_{timestamp:010.3f}"
            output_rows.append(
                {
                    "frame_id": frame_id,
                    "match_id": match_id,
                    "video_path": str(video_path).replace("\\", "/"),
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "original_label": "pending",
                    "label": "pending",
                    "split": "v10_annotation_pool",
                    "image_path": "",
                    "lt_x": "",
                    "lt_y": "",
                    "rt_x": "",
                    "rt_y": "",
                    "lb_x": "",
                    "lb_y": "",
                    "rb_x": "",
                    "rb_y": "",
                    "annotation_status": "pending",
                    "annotator": "",
                    "notes": (
                        f"pool_{index:02d}; separados por pelo menos 10s "
                        "dos timestamps existentes"
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Frames preparados: {len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
