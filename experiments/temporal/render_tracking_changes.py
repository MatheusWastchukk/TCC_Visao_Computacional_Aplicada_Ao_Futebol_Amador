"""Gera overlays dos intervalos alterados entre dois trackers."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Renderiza mudanças entre trackers.")
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    changes = rows(args.changes)
    tracking = rows(args.tracking / "frame_tracking.csv")
    annotations = {row["interval_id"]: row for row in rows(args.annotations)}
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tracking:
        by_block[row["block_id"]].append(row)
    manifest: list[dict[str, str]] = []
    for change in changes:
        annotation = annotations[change["interval_id"]]
        candidates = [row for row in by_block[annotation["block_id"]] if float(annotation["start_seconds"]) <= float(row["timestamp_seconds"]) <= float(annotation["end_seconds"])]
        if not candidates:
            continue
        row = candidates[len(candidates) // 2]
        capture = cv2.VideoCapture(str(args.videos_dir / annotation["video_filename"]))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(row["timestamp_seconds"]) * 1000.0)
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            continue
        values = [row.get(f"track_{axis}") for axis in ("x1", "y1", "x2", "y2")]
        if all(value not in {None, ""} for value in values):
            x1, y1, x2, y2 = map(lambda value: int(float(value)), values)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(image, "TRACK EXP-0004", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, f"{change['interval_id']} | {change['base_result']} -> {change['candidate_result']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        target = args.output / "overlays" / f"{change['interval_id']}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image)
        manifest.append({**change, "timestamp_hhmmss": row["timestamp_hhmmss"], "overlay_path": target.as_posix()})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"overlays={len(manifest)}")
    print(f"manifest={args.output / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
