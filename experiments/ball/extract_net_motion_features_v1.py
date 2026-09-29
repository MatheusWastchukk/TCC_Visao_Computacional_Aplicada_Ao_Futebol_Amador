"""Extrai movimento local da rede/goleira de grupos já minerados."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extract_local_trajectory_features_v1 import temporal_groups
from src.events.net_motion_v1 import net_motion_metrics, stabilized_difference


def read_frame(capture: cv2.VideoCapture, timestamp: float):
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Não foi possível ler frame em {timestamp:.3f}s.")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai movimento local da goleira para grupos temporais.")
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-gap-seconds", type=float, default=0.2)
    parser.add_argument("--limit-groups", type=int, default=None, help="Limite diagnóstico; omitido processa todos os grupos.")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.frame_gap_seconds <= 0:
        parser.error("--frame-gap-seconds deve ser positivo.")
    frame_rows: list[dict[str, str]] = []
    for frames_path in args.frames:
        with frames_path.open(newline="", encoding="utf-8") as file:
            frame_rows.extend(csv.DictReader(file))
    groups = temporal_groups(frame_rows, 3.0, 2)
    if args.limit_groups is not None:
        groups = groups[:args.limit_groups]
    captures: dict[str, cv2.VideoCapture] = {}
    indexes: dict[str, int] = defaultdict(int)
    output_rows: list[dict[str, str]] = []
    try:
        for position, group in enumerate(groups, start=1):
            representative = max(group, key=lambda row: float(row["candidate_score"]))
            match_id, video_name = representative["match_id"], representative["video_filename"]
            indexes[match_id] += 1
            capture = captures.get(video_name)
            if capture is None:
                capture = cv2.VideoCapture(str(args.videos_dir / video_name))
                if not capture.isOpened():
                    raise FileNotFoundError(args.videos_dir / video_name)
                captures[video_name] = capture
            timestamp = float(representative["timestamp_seconds"])
            difference = stabilized_difference(read_frame(capture, timestamp - args.frame_gap_seconds), read_frame(capture, timestamp))
            goal_box = tuple(float(value) for value in json.loads(representative["goal_box_json"]))
            inside, ring, ratio = net_motion_metrics(difference, goal_box)
            output_rows.append({
                "group_uid": f"{match_id}::temporal_group_{indexes[match_id]:04d}", "match_id": match_id,
                "net_motion_inside_mean": f"{inside:.8f}", "net_motion_ring_mean": f"{ring:.8f}",
                "net_motion_ratio": f"{ratio:.8f}", "representative_timestamp_seconds": f"{timestamp:.6f}",
            })
            if position % args.progress_every == 0 or position == len(groups):
                print(f"progresso={position}/{len(groups)}", flush=True)
    finally:
        for capture in captures.values():
            capture.release()
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0]) if output_rows else ["group_uid", "match_id"]
    with (args.output / "net_motion_groups_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {"experiment_id": args.output.name, "scope": "diagnóstico TRAIN; não altera o V4", "groups": len(output_rows), "frame_gap_seconds": args.frame_gap_seconds}
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
