"""Prepara candidatos adicionais de treino sem alterar validação ou teste."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import cv2

from prepare_ball_dataset_candidates import FIELDS, extract_frame, format_time


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara candidatos adicionais da bola para treino.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-per-match", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    source_rows = list(csv.DictReader(args.source.open(newline="", encoding="utf-8")))
    manifest = {}
    for row in source_rows:
        if row["split"] == "train":
            manifest.setdefault(row["match_id"], row)
    used: dict[str, set[int]] = {}
    for row in source_rows:
        match_id = row["match_id"]
        match = re.search(r"frame_index=(\d+)", row.get("notes", ""))
        used.setdefault(match_id, set()).update({int(match.group(1))} if match else set())
    randomizer = random.Random(args.seed)
    output_rows: list[dict[str, str]] = []
    for match_id, manifest_row in manifest.items():
        video_path = args.videos_dir / manifest_row["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if frame_count < 2 or fps <= 0:
            raise ValueError(f"Metadados inválidos no vídeo: {video_path}")
        first_frame = max(1, round(frame_count * 0.05))
        last_frame = min(frame_count - 1, round(frame_count * 0.95))
        available = [frame for frame in range(first_frame, last_frame + 1) if frame not in used.get(match_id, set())]
        selected = sorted(randomizer.sample(available, args.extra_per_match))
        for index, frame_index in enumerate(selected, start=1):
            timestamp = frame_index / fps
            sample_id = f"{match_id}_ball_extra_{index:03d}"
            image_path = args.frames_dir / "train" / match_id / f"{sample_id}_{timestamp:010.3f}.jpg"
            extract_frame(args.videos_dir / manifest_row["video_filename"], frame_index, image_path)
            output_rows.append({
                "sample_id": sample_id, "match_id": match_id,
                "video_filename": manifest_row["video_filename"],
                "capture_domain": manifest_row["capture_domain"], "split": "train",
                "timestamp_seconds": f"{timestamp:.3f}", "timestamp_hhmmss": format_time(timestamp),
                "image_path": image_path.as_posix(), "ball_visibility": "pending",
                "ball_x1": "", "ball_y1": "", "ball_x2": "", "ball_y2": "",
                "annotation_status": "pending", "annotator": "",
                "notes": f"seed={args.seed}; frame_index={frame_index}; origem=extra_v2",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"extra_train={len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
