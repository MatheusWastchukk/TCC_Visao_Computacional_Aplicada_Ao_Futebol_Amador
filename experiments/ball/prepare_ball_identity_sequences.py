"""Extrai sequências difíceis do treino para anotação temporal de identidade."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


FIELDS = [
    "sequence_id", "sequence_frame", "sample_id", "match_id", "video_filename", "capture_domain",
    "split", "timestamp_seconds", "timestamp_hhmmss", "image_path", "ball_visibility",
    "ball_of_play_box_json", "distractor_boxes_json", "annotation_status", "annotator", "notes",
]


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara sequências temporais de identidade da bola no treino.")
    parser.add_argument("--hard-examples", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=10)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.4, -0.2, 0.0, 0.2, 0.4])
    args = parser.parse_args()

    with args.hard_examples.open(newline="", encoding="utf-8") as file:
        hard_rows = list(csv.DictReader(file))
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hard_rows:
        if row["split"] == "train" and row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            by_match[row["match_id"]].append(row)
    selected_matches = sorted(by_match, key=lambda match_id: (-len(by_match[match_id]), match_id))[:args.contexts]
    centers = [
        min(by_match[match_id], key=lambda row: (float(row["mined_iou"]), row["sample_id"]))
        for match_id in selected_matches
    ]
    if not centers:
        raise ValueError("Nenhum contexto difícil de treino foi encontrado.")

    output_rows: list[dict[str, str]] = []
    for sequence_number, center in enumerate(centers, start=1):
        video_path = args.videos_dir / center["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            center_seconds = float(center["timestamp_seconds"])
            sequence_id = f"identity_train_sequence_{sequence_number:02d}"
            for frame_number, offset in enumerate(args.offsets, start=1):
                seconds = min(max(0.0, center_seconds + offset), (frame_count - 1) / fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Não foi possível ler {video_path} em {seconds:.3f} s.")
                sample_id = f"{sequence_id}_frame_{frame_number:02d}"
                image_path = args.frames_dir / center["match_id"] / f"{sample_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Não foi possível salvar {image_path}.")
                output_rows.append({
                    "sequence_id": sequence_id,
                    "sequence_frame": str(frame_number),
                    "sample_id": sample_id,
                    "match_id": center["match_id"],
                    "video_filename": center["video_filename"],
                    "capture_domain": center["capture_domain"],
                    "split": "train",
                    "timestamp_seconds": f"{seconds:.3f}",
                    "timestamp_hhmmss": format_time(seconds),
                    "image_path": image_path.as_posix(),
                    "ball_visibility": "VISIBLE",
                    "ball_of_play_box_json": "",
                    "distractor_boxes_json": "[]",
                    "annotation_status": "pending",
                    "annotator": "",
                    "notes": f"origem={center['sample_id']}; delta_seconds={offset:+.1f}; uso=identidade_temporal_treino",
                })
        finally:
            capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"sequencias={len(centers)}")
    print(f"frames={len(output_rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
