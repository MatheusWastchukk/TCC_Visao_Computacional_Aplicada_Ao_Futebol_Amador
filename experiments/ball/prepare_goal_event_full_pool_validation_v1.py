"""Apply the frozen TRAIN temporal ranker to every VALIDATION temporal group."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frames / fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows, fields = read_csv(args.features)
    manifest, manifest_fields = read_csv(args.split_manifest)
    if not rows or {"group_uid", "match_id", "best_timestamp_seconds"} - set(fields):
        raise ValueError("VALIDATION temporal features lack group, match, or timestamp fields.")
    if {"match_id", "video_filename", "split"} - set(manifest_fields):
        raise ValueError("Split manifest is incomplete.")
    validation_videos = {r["match_id"]: r["video_filename"] for r in manifest if r["split"].lower() == "validation"}
    if not validation_videos or any(r["match_id"] not in validation_videos for r in rows):
        raise AssertionError("Feature rows must belong exclusively to manifest VALIDATION matches.")
    if len({r["group_uid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate group_uid in the full VALIDATION pool.")

    frozen = joblib.load(args.ranker)
    features = frozen.get("features")
    model = frozen.get("model")
    if not features or model is None or set(features) - set(fields):
        raise ValueError("Frozen temporal ranker feature contract does not match input dataset.")
    durations = {m: duration(args.videos_dir / filename) for m, filename in validation_videos.items() if any(r["match_id"] == m for r in rows)}
    x = np.asarray([[float(row[feature]) for feature in features] for row in rows], dtype=np.float32)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Frozen ranker has no positive class.")
    scores = model.predict_proba(x)[:, classes.index(1)]

    output_rows: list[dict[str, object]] = []
    for source, score in zip(rows, scores):
        match = source["match_id"]
        center = float(source["best_timestamp_seconds"])
        output_rows.append({
            "event_id": source["group_uid"],
            "group_uid": source["group_uid"],
            "match_id": match,
            "video_filename": validation_videos[match],
            "split": "validation",
            "event_timestamp_seconds": f"{center:.6f}",
            "rank_score": f"{float(score):.8f}",
            "candidate_start_seconds": f"{center - 10.0:.6f}",
            "candidate_end_seconds": f"{center + 20.0:.6f}",
            "delivery_start_seconds": f"{max(0.0, center - 20.0):.6f}",
            "delivery_end_seconds": f"{min(durations[match], center + 40.0):.6f}",
        })

    args.output.mkdir(parents=True)
    with (args.output / "full_pool_temporal_candidates_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    counts = Counter(row["match_id"] for row in output_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "all temporal groups from VALIDATION; ranker loaded from TRAIN-only frozen artifact",
        "groups": len(output_rows),
        "matches": len(counts),
        "groups_per_match": dict(sorted(counts.items())),
        "ranker": str(args.ranker),
        "features_used": features,
        "candidate_window": "10 seconds before and 20 seconds after best timestamp; visual scorer bounds to video duration",
        "delivery_window": "20 seconds before and 40 seconds after best timestamp; truncated at video bounds",
        "warning": "No VALIDATION labels used to fit or select the ranker; TEST not accessed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
