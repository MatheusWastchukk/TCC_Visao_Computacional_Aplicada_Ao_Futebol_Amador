"""Seleciona e extrai frames aleatórios das partidas de treinamento."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2


FIELDS = [
    "frame_id",
    "match_id",
    "video_filename",
    "timestamp_seconds",
    "original_label",
    "label",
    "split",
    "image_path",
    "lt_x",
    "lt_y",
    "rt_x",
    "rt_y",
    "lb_x",
    "lb_y",
    "rb_x",
    "rb_y",
    "annotation_status",
    "annotator",
    "notes",
]


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
    parser = argparse.ArgumentParser(
        description="Extrai frames aleatórios para anotação manual da estrutura do gol."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-match", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--name-prefix", default="random")
    parser.add_argument(
        "--match-ids",
        help="Lista opcional de partidas train separadas por vírgula.",
    )
    args = parser.parse_args()
    if args.per_match <= 0:
        raise ValueError("per-match deve ser maior que zero")

    with args.manifest.open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    metadata = {item["match_id"]: item for item in validation["metadata"]}
    train_rows = [row for row in manifest if row["split"] == args.split]
    if args.match_ids:
        requested_matches = {item.strip() for item in args.match_ids.split(",") if item.strip()}
        train_rows = [row for row in train_rows if row["match_id"] in requested_matches]
        missing_matches = requested_matches - {row["match_id"] for row in train_rows}
        if missing_matches:
            raise ValueError(f"Partidas ausentes ou fora de {args.split}: {sorted(missing_matches)}")
    randomizer = random.Random(args.seed)
    output_rows: list[dict[str, str]] = []

    for row in train_rows:
        item = metadata[row["match_id"]]
        frame_count = int(item["frame_count"])
        fps = float(item["fps"])
        first_frame = max(1, round(frame_count * 0.05))
        last_frame = min(frame_count - 1, round(frame_count * 0.95))
        frame_indices = randomizer.sample(range(first_frame, last_frame + 1), args.per_match)
        for sample_index, frame_index in enumerate(sorted(frame_indices), start=1):
            timestamp = frame_index / fps
            frame_id = f"{row['match_id']}_{args.name_prefix}_{sample_index:02d}"
            relative_image_path = (
                args.frames_dir / row["match_id"] / f"{frame_id}_{timestamp:010.3f}.jpg"
            )
            extract_frame(args.videos_dir / row["video_filename"], frame_index, relative_image_path)
            output_rows.append(
                {
                    "frame_id": frame_id,
                    "match_id": row["match_id"],
                    "video_filename": row["video_filename"],
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "original_label": "pending",
                    "label": "pending",
                    "split": args.split,
                    "image_path": relative_image_path.as_posix(),
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
                    "notes": f"seed={args.seed}; frame_index={frame_index}",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"partidas_{args.split}={len(train_rows)}")
    print(f"frames_extraidos={len(output_rows)}")
    print(f"seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
