"""Extrai frames ao redor de eventos catalogados nas partidas train."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


FIELDS = [
    "frame_id", "match_id", "video_filename", "timestamp_seconds", "original_label",
    "label", "split", "image_path", "lt_x", "lt_y", "rt_x", "rt_y", "lb_x", "lb_y",
    "rb_x", "rb_y", "annotation_status", "annotator", "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai frames dirigidos para anotação da goleira.")
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.timestamps.open(newline="", encoding="utf-8") as file:
        timestamp_rows = list(csv.DictReader(file))
    with args.manifest.open(newline="", encoding="utf-8") as file:
        manifest = {row["match_id"]: row for row in csv.DictReader(file)}

    output_rows: list[dict[str, str]] = []
    for row in timestamp_rows:
        match = manifest[row["match_id"]]
        if match["split"] != "train":
            raise ValueError(f"Amostra fora de train: {row['match_id']}")
        timestamp = float(row["event_timestamp_seconds"]) + float(row["offset_seconds"])
        if timestamp < 0:
            raise ValueError(f"Timestamp negativo: {row}")
        capture = cv2.VideoCapture(str(args.videos_dir / match["video_filename"]))
        if not capture.isOpened():
            raise ValueError(f"Não foi possível abrir: {match['video_filename']}")
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise ValueError(f"Não foi possível extrair {timestamp:.3f}s de {match['video_filename']}")

        frame_id = f"{row['match_id']}_{row['event_category']}_{row['offset_seconds'].replace('-', 'm').replace('.', '_')}"
        image_path = args.frames_dir / row["match_id"] / f"{frame_id}_{timestamp:010.3f}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), frame):
            raise ValueError(f"Não foi possível gravar: {image_path}")
        output_rows.append(
            {
                "frame_id": frame_id,
                "match_id": row["match_id"],
                "video_filename": match["video_filename"],
                "timestamp_seconds": f"{timestamp:.3f}",
                "original_label": "pending",
                "label": "pending",
                "split": "train",
                "image_path": image_path.as_posix(),
                "lt_x": "", "lt_y": "", "rt_x": "", "rt_y": "",
                "lb_x": "", "lb_y": "", "rb_x": "", "rb_y": "",
                "annotation_status": "pending",
                "annotator": "",
                "notes": f"categoria={row['event_category']}; offset={row['offset_seconds']}s; {row['notes']}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"frames_extraidos={len(output_rows)}")
    print(f"partidas={len({row['match_id'] for row in output_rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
