"""Agrupa e renderiza sequências em que a bola entra na caixa de uma goleira."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def draw_box(image, values: list[str], color, label: str) -> None:
    x1, y1, x2, y2 = (int(float(value)) for value in values)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    cv2.putText(image, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara revisão visual bola × goleira.")
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap-seconds", type=float, default=0.4)
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Mantém somente sequências com esta quantidade mínima de frames amostrados.",
    )
    args = parser.parse_args()
    rows = [row for row in read_csv(args.relations) if row["ball_inside_nearest_goal"] == "True"]
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_block[row["block_id"]].append(row)
    sequences: list[list[dict[str, str]]] = []
    for block_rows in by_block.values():
        ordered = sorted(block_rows, key=lambda row: float(row["timestamp_seconds"]))
        current: list[dict[str, str]] = []
        previous_time: float | None = None
        for row in ordered:
            timestamp = float(row["timestamp_seconds"])
            if previous_time is not None and timestamp - previous_time > args.max_gap_seconds:
                sequences.append(current)
                current = []
            current.append(row)
            previous_time = timestamp
        if current:
            sequences.append(current)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for number, sequence in enumerate(sequences, start=1):
        if len(sequence) < args.min_frames:
            continue
        row = sequence[len(sequence) // 2]
        capture = cv2.VideoCapture(str(args.videos_dir / row["video_filename"]))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(row["timestamp_seconds"]) * 1000.0)
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            continue
        draw_box(image, [row[f"ball_{axis}"] for axis in ("x1", "y1", "x2", "y2")], (255, 0, 255), "BOLA TRACK")
        draw_box(image, [row[f"nearest_goal_{axis}"] for axis in ("x1", "y1", "x2", "y2")], (0, 200, 0), "GOAL STRUCTURE")
        start, end = sequence[0]["timestamp_hhmmss"], sequence[-1]["timestamp_hhmmss"]
        cv2.putText(image, f"sequencia {number} | {start} a {end} | frames={len(sequence)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        target = args.output / "overlays" / f"ball_goal_sequence_{number:03d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image)
        manifest.append({
            "sequence_id": f"ball_goal_sequence_{number:03d}", "block_id": row["block_id"], "match_id": row["match_id"],
            "video_filename": row["video_filename"], "start_timestamp_hhmmss": start, "end_timestamp_hhmmss": end,
            "frames_inside_goal_box": str(len(sequence)), "representative_timestamp_hhmmss": row["timestamp_hhmmss"],
            "overlay_path": target.as_posix(), "review_status": "pending", "review_label": "", "review_notes": "",
        })
    if not manifest:
        raise ValueError("Nenhuma sequência atende aos filtros de revisão solicitados.")
    with (args.output / "review_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"frames_inside={len(rows)}")
    print(f"sequencias={len(manifest)}")
    print(f"manifest={args.output / 'review_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
