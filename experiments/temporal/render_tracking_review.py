"""Gera overlays representativos para revisão visual dos tracks."""

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
    parser = argparse.ArgumentParser(description="Gera overlays de revisão do tracking.")
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    frames = read_csv(args.tracking / "frame_tracking.csv")
    annotations = [row for row in read_csv(args.annotations) if row.get("annotation_status") == "completed"]
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frames:
        by_block[row["block_id"]].append(row)
    output_rows: list[dict[str, str]] = []
    for annotation in annotations:
        rows = [row for row in by_block[annotation["block_id"]] if float(annotation["start_seconds"]) <= float(row["timestamp_seconds"]) <= float(annotation["end_seconds"])]
        if not rows:
            continue
        representative = rows[len(rows) // 2]
        video_path = args.videos_dir / annotation["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(representative["timestamp_seconds"]) * 1000.0)
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            continue
        active = sum(row["track_active"] == "True" for row in rows)
        rate = active / len(rows)
        confirmed = rate >= args.threshold
        label = annotation["interval_label"]
        if label == "goal_structure":
            result = "TP" if confirmed else "FN"
        elif label == "not_goal":
            result = "FP" if confirmed else "TN"
        else:
            result = "INCONCLUSIVO"
        x_values = [representative.get(f"track_{axis}") for axis in ("x1", "y1", "x2", "y2")]
        if all(value not in {None, ""} for value in x_values):
            x1, y1, x2, y2 = map(lambda value: int(float(value)), x_values)
            color = (0, 0, 255) if result in {"FP", "FN"} else (0, 200, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(image, "TRACK", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"{result} | real={label} | aparicao={rate:.1%}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(image, f"{annotation['interval_id']} | {representative['timestamp_hhmmss']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        target = args.output / "overlays" / f"{annotation['interval_id']}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image)
        output_rows.append({"interval_id": annotation["interval_id"], "match_id": annotation["match_id"], "interval_label": label, "result": result, "appearance_rate": f"{rate:.6f}", "timestamp_hhmmss": representative["timestamp_hhmmss"], "overlay_path": target.as_posix()})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "review_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"overlays={len(output_rows)}")
    print(f"manifest={args.output / 'review_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
