"""Prepara candidatos de frames para anotação da bola, separados por partida."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2


FIELDS = [
    "sample_id",
    "match_id",
    "video_filename",
    "capture_domain",
    "split",
    "timestamp_seconds",
    "timestamp_hhmmss",
    "image_path",
    "ball_visibility",
    "ball_x1",
    "ball_y1",
    "ball_x2",
    "ball_y2",
    "annotation_status",
    "annotator",
    "notes",
]


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Não foi possível ler o frame {frame_index}: {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise ValueError(f"Não foi possível gravar o frame: {output_path}")
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara frames candidatos para anotação da bola.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-match", type=int, default=20)
    parser.add_argument("--validation-per-match", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))["metadata"]
    metadata_by_match = {row["match_id"]: row for row in metadata}
    randomizer = random.Random(args.seed)
    output_rows: list[dict[str, str]] = []

    for manifest_row in manifest:
        split = manifest_row["split"]
        if split == "train":
            sample_count = args.train_per_match
        elif split == "validation":
            sample_count = args.validation_per_match
        else:
            continue
        metadata_row = metadata_by_match[manifest_row["match_id"]]
        frame_count = int(metadata_row["frame_count"])
        fps = float(metadata_row["fps"])
        first_frame = max(1, round(frame_count * 0.05))
        last_frame = min(frame_count - 1, round(frame_count * 0.95))
        if sample_count > last_frame - first_frame + 1:
            raise ValueError(f"Amostras demais para {manifest_row['match_id']}")
        selected = sorted(randomizer.sample(range(first_frame, last_frame + 1), sample_count))
        for index, frame_index in enumerate(selected, start=1):
            timestamp = frame_index / fps
            sample_id = f"{manifest_row['match_id']}_ball_{index:03d}"
            image_path = args.frames_dir / split / manifest_row["match_id"] / f"{sample_id}_{timestamp:010.3f}.jpg"
            extract_frame(args.videos_dir / manifest_row["video_filename"], frame_index, image_path)
            output_rows.append(
                {
                    "sample_id": sample_id,
                    "match_id": manifest_row["match_id"],
                    "video_filename": manifest_row["video_filename"],
                    "capture_domain": manifest_row["capture_domain"],
                    "split": split,
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "timestamp_hhmmss": format_time(timestamp),
                    "image_path": image_path.as_posix(),
                    "ball_visibility": "pending",
                    "ball_x1": "",
                    "ball_y1": "",
                    "ball_x2": "",
                    "ball_y2": "",
                    "annotation_status": "pending",
                    "annotator": "",
                    "notes": f"seed={args.seed}; frame_index={frame_index}",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"train={sum(row['split'] == 'train' for row in output_rows)}")
    print(f"validation={sum(row['split'] == 'validation' for row in output_rows)}")
    print(f"total={len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
