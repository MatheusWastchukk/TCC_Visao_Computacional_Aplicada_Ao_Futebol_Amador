"""Localiza candidatos de mudança visual no recorte do placar fechado."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def preprocess(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, (128, 64), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def read_series(video_path: Path, roi: list[int], sample_seconds: float) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, round(fps * sample_seconds))
        x, y, width, height = roi
        rows: list[dict[str, object]] = []
        frame_number = 0
        previous: np.ndarray | None = None
        while frame_number < total:
            ok, image = capture.read()
            if not ok:
                break
            if y + height > image.shape[0] or x + width > image.shape[1]:
                raise ValueError(f"ROI fora do frame em {video_path}: {roi}")
            feature = preprocess(image[y : y + height, x : x + width])
            difference = 0.0 if previous is None else float(cv2.absdiff(feature, previous).mean() / 255.0)
            rows.append({"timestamp_seconds": frame_number / fps, "difference": difference})
            previous = feature
            remaining = min(stride - 1, total - frame_number - 1)
            for _ in range(remaining):
                if not capture.grab():
                    break
            frame_number += stride
        return rows
    finally:
        capture.release()


def robust_threshold(differences: list[float], multiplier: float, floor: float) -> float:
    values = np.asarray(differences[1:] or [0.0], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(floor, median + multiplier * 1.4826 * mad)


def main() -> int:
    parser = argparse.ArgumentParser(description="Varre placares fechados e gera candidatos de mudança visual para revisão manual.")
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--threshold-mad-multiplier", type=float, default=8.0)
    parser.add_argument("--threshold-floor", type=float, default=0.012)
    parser.add_argument("--cooldown-seconds", type=float, default=45.0)
    args = parser.parse_args()
    with args.rois.open(newline="", encoding="utf-8") as file:
        matches = [row for row in csv.DictReader(file) if row["annotation_status"] == "completed" and row["scoreboard_roi_json"]]
    if not matches:
        raise ValueError("Não há ROIs concluídas.")
    series_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for match in matches:
        series = read_series(args.videos_dir / match["video_filename"], json.loads(match["scoreboard_roi_json"]), args.sample_seconds)
        threshold = robust_threshold([float(row["difference"]) for row in series], args.threshold_mad_multiplier, args.threshold_floor)
        last_candidate = -float("inf")
        count = 0
        for row in series:
            timestamp = float(row["timestamp_seconds"])
            is_candidate = float(row["difference"]) >= threshold and timestamp - last_candidate >= args.cooldown_seconds
            series_rows.append({"match_id": match["match_id"], "timestamp_seconds": f"{timestamp:.3f}", "difference": f"{float(row['difference']):.8f}", "threshold": f"{threshold:.8f}", "is_candidate": str(is_candidate)})
            if is_candidate:
                count += 1; last_candidate = timestamp
                candidate_rows.append({"candidate_id": f"score_change_{match['match_id']}_{count:03d}", "match_id": match["match_id"], "video_filename": match["video_filename"], "timestamp_seconds": f"{timestamp:.3f}", "expected_goals_registered": match["expected_goals"], "difference": f"{float(row['difference']):.8f}", "threshold": f"{threshold:.8f}", "manual_label": "", "annotation_status": "pending", "notes": "Candidato por diferença visual no placar; revisão humana obrigatória."})
        summaries.append({"match_id": match["match_id"], "expected_goals_registered": match["expected_goals"], "samples": len(series), "threshold": threshold, "score_change_candidates": count})
    args.output.mkdir(parents=True, exist_ok=True)
    def write(name: str, rows: list[dict[str, object]]) -> None:
        with (args.output / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write("scoreboard_difference_timeseries.csv", series_rows)
    write("score_change_candidates.csv", candidate_rows)
    write("scoreboard_scan_summary.csv", summaries)
    print(json.dumps({"experiment_id": args.output.name, "matches": len(matches), "candidates": len(candidate_rows), "summary": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
