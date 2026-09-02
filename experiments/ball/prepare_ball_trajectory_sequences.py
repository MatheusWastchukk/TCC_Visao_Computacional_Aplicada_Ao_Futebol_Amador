"""Extrai clipes contínuos de desenvolvimento para anotar trajetórias da bola."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from prepare_ball_identity_sequences import FIELDS, format_time


def spaced(items: list[str], count: int) -> list[str]:
    """Seleciona partidas distribuídas para reduzir concentração de contexto."""
    if count >= len(items):
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara clipes de desenvolvimento para anotar trajetórias da bola.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=3)
    parser.add_argument("--seconds-before", type=float, default=2.0)
    parser.add_argument("--seconds-after", type=float, default=2.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps deve ser maior que zero.")

    with args.annotations.open(newline="", encoding="utf-8") as file:
        annotations = list(csv.DictReader(file))
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotations:
        if row["split"] == "train" and row["annotation_status"] == "completed" and row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            by_match[row["match_id"]].append(row)
    matches = spaced(sorted(by_match), args.contexts)
    if not matches:
        raise ValueError("Não há positivos anotados no split train para formar clipes.")

    offsets = []
    value = -args.seconds_before
    step = 1.0 / args.sample_fps
    while value <= args.seconds_after + 1e-9:
        offsets.append(value)
        value += step
    output_rows: list[dict[str, str]] = []
    for sequence_number, match_id in enumerate(matches, start=1):
        candidates = sorted(by_match[match_id], key=lambda row: float(row["timestamp_seconds"]))
        center = candidates[len(candidates) // 2]
        video_path = args.videos_dir / center["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            sequence_id = f"trajectory_train_sequence_{sequence_number:02d}"
            for frame_number, offset in enumerate(offsets, start=1):
                seconds = min(max(0.0, float(center["timestamp_seconds"]) + offset), (frame_count - 1) / video_fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * video_fps))
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Não foi possível ler {video_path} em {seconds:.3f} s.")
                sample_id = f"{sequence_id}_frame_{frame_number:03d}"
                image_path = args.frames_dir / match_id / f"{sample_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Não foi possível salvar {image_path}.")
                output_rows.append({
                    "sequence_id": sequence_id, "sequence_frame": str(frame_number), "sample_id": sample_id,
                    "match_id": match_id, "video_filename": center["video_filename"], "capture_domain": center["capture_domain"],
                    "split": "train", "timestamp_seconds": f"{seconds:.3f}", "timestamp_hhmmss": format_time(seconds),
                    "image_path": image_path.as_posix(), "ball_visibility": "VISIBLE", "ball_of_play_box_json": "",
                    "distractor_boxes_json": "[]", "annotation_status": "pending", "annotator": "",
                    "notes": f"origem={center['sample_id']}; delta_seconds={offset:+.1f}; uso=trajetoria_desenvolvimento",
                })
        finally:
            capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"sequências={len(matches)}")
    print(f"frames={len(output_rows)}")
    print(f"partidas={','.join(matches)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
