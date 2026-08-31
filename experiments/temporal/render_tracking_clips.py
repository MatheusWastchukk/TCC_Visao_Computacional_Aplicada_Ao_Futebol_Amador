"""Renderiza clipes dos intervalos inconclusivos da revisão do tracking."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera clipes para revisar intervalos inconclusivos.")
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = read_csv(args.tracking / "frame_tracking.csv")
    review = [row for row in read_csv(args.review) if row.get("review_label") == "INCONCLUSIVO"]
    annotations = {row["interval_id"]: row for row in read_csv(args.annotations)}
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frames:
        by_block[row["block_id"]].append(row)
    manifest: list[dict[str, str]] = []
    for review_row in review:
        annotation = annotations[review_row["interval_id"]]
        frame_rows = [row for row in by_block[annotation["block_id"]] if float(annotation["start_seconds"]) <= float(row["timestamp_seconds"]) <= float(annotation["end_seconds"])]
        if not frame_rows:
            continue
        video_path = args.videos_dir / annotation["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 5.0
        target = args.output / f"{annotation['interval_id']}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for row in frame_rows:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(row["timestamp_seconds"]) * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            values = [row.get(f"track_{axis}") for axis in ("x1", "y1", "x2", "y2")]
            if all(value not in {None, ""} for value in values):
                x1, y1, x2, y2 = map(lambda value: int(float(value)), values)
                color = (0, 200, 0) if row["detected"] == "True" else (0, 165, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
                cv2.putText(image, "TRACK", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(image, f"{annotation['interval_id']} | {row['timestamp_hhmmss']} | active={row['track_active']} | gap={row['gap']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            writer.write(image)
        writer.release()
        capture.release()
        manifest.append({"interval_id": annotation["interval_id"], "match_id": annotation["match_id"], "video_filename": annotation["video_filename"], "clip_path": target.as_posix(), "review_label": review_row["review_label"]})
    with (args.output / "clip_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"clips={len(manifest)}")
    print(f"manifest={args.output / 'clip_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
