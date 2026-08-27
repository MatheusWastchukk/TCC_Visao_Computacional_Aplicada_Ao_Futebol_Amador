"""Prepara candidatos de sequências contínuas para o piloto temporal V10."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara candidatos de sequências de 3 a 5 segundos para validação temporal."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--positive-count", type=int, default=8)
    parser.add_argument("--negative-count", type=int, default=7)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def format_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def main() -> int:
    args = parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))

    by_image: dict[str, dict[str, str]] = {}
    for row in source_rows:
        image_path = row["image_path"]
        current = by_image.get(image_path)
        if current is None or row["label"] == "goal_structure":
            by_image[image_path] = row

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in by_image.values():
        if row["label"] in {"goal_structure", "not_goal"}:
            groups[row["label"]].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for label, count in (("goal_structure", args.positive_count), ("not_goal", args.negative_count)):
        candidates = groups[label].copy()
        rng.shuffle(candidates)
        selected.extend(candidates[:count])

    half_window = args.window_seconds / 2.0
    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        timestamp = float(row["timestamp_seconds"])
        video_path = args.videos_dir / row["video_filename"]
        output_rows.append(
            {
                "sequence_id": f"temporal_v10_candidate_{index:03d}",
                "match_id": row["match_id"],
                "video_path": str(video_path),
                "anchor_frame_id": row["frame_id"],
                "anchor_timestamp_seconds": f"{timestamp:.3f}",
                "start_seconds": f"{max(0.0, timestamp - half_window):.3f}",
                "end_seconds": f"{timestamp + half_window:.3f}",
                "anchor_time_hhmmss": format_time(timestamp),
                "start_time_hhmmss": format_time(timestamp - half_window),
                "end_time_hhmmss": format_time(timestamp + half_window),
                "true_label": "pending",
                "anchor_label": row["label"],
                "fps": f"{args.fps:g}",
                "annotation_status": "pending",
                "notes": "Confirmar no vídeo se o intervalo contém estrutura de gol de forma contínua; ajustar início/fim se necessário.",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"sequencias_criadas={len(output_rows)}")
    print(f"positivas_candidatas={sum(row['anchor_label'] == 'goal_structure' for row in output_rows)}")
    print(f"negativas_candidatas={sum(row['anchor_label'] == 'not_goal' for row in output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
