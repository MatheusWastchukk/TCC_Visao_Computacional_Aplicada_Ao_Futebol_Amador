"""Valida a divisão congelada e lê metadados básicos dos vídeos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


EXPECTED_COUNTS = {"train": 25, "validation": 8, "test": 9}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida o manifesto dataset_split_v1 e os contêineres de vídeo."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    errors: list[str] = []
    match_ids = [row["match_id"] for row in rows]
    if len(match_ids) != len(set(match_ids)):
        errors.append("match_id duplicado no manifesto")
    counts = {split: sum(row["split"] == split for row in rows) for split in EXPECTED_COUNTS}
    for split, expected in EXPECTED_COUNTS.items():
        if counts[split] != expected:
            errors.append(f"split {split}: esperado {expected}, encontrado {counts[split]}")

    metadata: list[dict[str, object]] = []
    for row in rows:
        video_path = args.videos_dir / row["video_filename"]
        if not video_path.is_file():
            errors.append(f"arquivo ausente: {video_path}")
            continue
        capture = cv2.VideoCapture(str(video_path))
        opened = bool(capture.isOpened())
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        capture.release()
        if not opened or frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            errors.append(f"contêiner não legível: {video_path.name}")
        metadata.append(
            {
                "match_id": row["match_id"],
                "video_filename": row["video_filename"],
                "split": row["split"],
                "capture_domain": row["capture_domain"],
                "file_size_bytes": video_path.stat().st_size,
                "frame_count": frame_count,
                "fps": round(fps, 6),
                "width": width,
                "height": height,
                "container_readable": opened,
            }
        )

    report = {
        "version": "dataset_split_v1",
        "manifest_rows": len(rows),
        "split_counts": counts,
        "video_files_checked": len(metadata),
        "readable_videos": sum(bool(item["container_readable"]) for item in metadata),
        "errors": errors,
        "status": "valid" if not errors else "invalid",
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    print(f"videos_verificados={len(metadata)}")
    print(f"videos_legiveis={report['readable_videos']}")
    if errors:
        for error in errors:
            print(f"erro={error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
