"""Amplia o conjunto de reinício com frames de partidas separadas."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2


def hhmmss(seconds: float) -> str:
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def duration_seconds(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"Metadados inválidos para {path}.")
    return frames / fps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cria V2 de reinício com amostras aleatórias de partidas fora do piloto inicial."
    )
    parser.add_argument("--base-samples", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-video", type=int, default=6)
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--margin-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.samples_per_video < 1 or args.max_videos < 1:
        raise ValueError("As quantidades de vídeos e amostras devem ser positivas.")
    with args.base_samples.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fields = list(rows[0])
    source_videos = {row["video_filename"] for row in rows}
    available = [path for path in sorted(args.videos_dir.glob("*.mp4")) if path.name not in source_videos]
    if len(available) < args.max_videos:
        raise ValueError(f"Há somente {len(available)} partidas separadas disponíveis.")

    rng = random.Random(args.seed)
    for path in available[: args.max_videos]:
        duration = duration_seconds(path)
        low = min(args.margin_seconds, duration / 4)
        high = max(low, duration - low)
        for _ in range(args.samples_per_video):
            timestamp = rng.uniform(low, high) if high > low else duration / 2
            rows.append(
                {
                    "sample_id": f"restart_v2_{len(rows) + 1:03d}",
                    "context_id": "",
                    "event_id": "",
                    "match_id": path.stem,
                    "video_filename": path.name,
                    "sample_timestamp_seconds": f"{timestamp:.3f}",
                    "sample_timestamp_hhmmss": hhmmss(timestamp),
                    "sampling_group": "AMOSTRA_ALEATORIA_PARTIDA_SEPARADA",
                    "reference_context_decision": "SEM_REFERENCIA",
                    "restart_label": "",
                    "annotation_status": "pending",
                    "notes": "Amostra aleatória cega de partida fora do piloto inicial.",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"base={len(rows) - args.max_videos * args.samples_per_video}")
    print(f"novas={args.max_videos * args.samples_per_video}")
    print(f"total={len(rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
