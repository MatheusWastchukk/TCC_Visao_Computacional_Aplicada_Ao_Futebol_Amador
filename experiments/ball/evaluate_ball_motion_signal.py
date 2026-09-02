"""Avalia se a bola anotada possui sinal de movimento após estabilização simples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_frame(capture: cv2.VideoCapture, seconds: float) -> np.ndarray:
    """Lê um frame em segundos absolutos do vídeo."""
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Não foi possível ler o frame em {seconds:.3f} s.")
    return frame


def stabilized_difference(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Remove movimento global aproximado da câmera antes da diferença absoluta."""
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=600, qualityLevel=0.01, minDistance=8)
    if points is not None:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
        if next_points is not None and status is not None:
            source = points[status.ravel() == 1]
            target = next_points[status.ravel() == 1]
            if len(source) >= 8:
                matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
                if matrix is not None:
                    previous_gray = cv2.warpAffine(
                        previous_gray,
                        matrix,
                        (current_gray.shape[1], current_gray.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
    return cv2.absdiff(previous_gray, current_gray)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mede o sinal temporal de movimento na caixa da bola anotada.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-gap-seconds", type=float, default=0.2)
    args = parser.parse_args()

    annotations = [
        row for row in read_rows(args.annotations) if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}
    ]
    identity = {row["ambiguous_id"]: row for row in read_rows(args.identity_manifest)}
    relation_rows = read_rows(args.relations)
    relations = {(row["block_id"], row["timestamp_hhmmss"]): row for row in relation_rows}
    captures: dict[str, cv2.VideoCapture] = {}
    output_rows: list[dict[str, str]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for annotation in annotations:
            ambiguous_id = annotation["sample_id"].removeprefix("candidate_recall_")
            source = identity.get(ambiguous_id)
            if source is None:
                raise KeyError(f"Identidade não encontrada: {ambiguous_id}")
            relation = relations.get((source["block_id"], source["timestamp_hhmmss"]))
            if relation is None:
                raise KeyError(f"Relação não encontrada: {ambiguous_id}")
            video_name = source["video_filename"]
            capture = captures.get(video_name)
            if capture is None:
                capture = cv2.VideoCapture(str(args.videos_dir / video_name))
                if not capture.isOpened():
                    raise FileNotFoundError(args.videos_dir / video_name)
                captures[video_name] = capture
            timestamp = float(relation["timestamp_seconds"])
            current = load_frame(capture, timestamp)
            previous = load_frame(capture, timestamp - args.frame_gap_seconds)
            difference = stabilized_difference(previous, current)
            x1, y1, x2, y2 = (int(float(annotation[field])) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"))
            x1, x2 = max(0, x1), min(difference.shape[1], x2)
            y1, y2 = max(0, y1), min(difference.shape[0], y2)
            mask = np.ones(difference.shape, dtype=bool)
            mask[y1:y2, x1:x2] = False
            inside = difference[y1:y2, x1:x2]
            background = difference[mask]
            inside_mean = float(np.mean(inside)) if inside.size else 0.0
            background_mean = float(np.mean(background)) if background.size else 0.0
            percentile = float(np.mean(background <= inside_mean)) if background.size else 0.0
            heatmap = cv2.applyColorMap(cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(current, 0.62, heatmap, 0.38, 0.0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(overlay, f"movimento={inside_mean:.1f} | percentil fundo={percentile:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            overlay_path = args.output / "overlays" / f"{annotation['sample_id']}.jpg"
            overlay_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
            output_rows.append({
                "sample_id": annotation["sample_id"],
                "match_id": annotation["match_id"],
                "timestamp_hhmmss": annotation["timestamp_hhmmss"],
                "ball_visibility": annotation["ball_visibility"],
                "motion_inside_mean": f"{inside_mean:.6f}",
                "motion_background_mean": f"{background_mean:.6f}",
                "motion_ratio": f"{inside_mean / max(background_mean, 1e-6):.6f}",
                "motion_background_percentile": f"{percentile:.6f}",
                "overlay_path": overlay_path.as_posix(),
            })
    finally:
        for capture in captures.values():
            capture.release()
    with (args.output / "motion_signal_frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    ratios = [float(row["motion_ratio"]) for row in output_rows]
    percentiles = [float(row["motion_background_percentile"]) for row in output_rows]
    summary = {
        "frames": len(output_rows),
        "mean_motion_ratio": sum(ratios) / len(ratios),
        "mean_background_percentile": sum(percentiles) / len(percentiles),
        "frames_above_background_mean": sum(ratio > 1.0 for ratio in ratios),
        "frames_above_background_p75": sum(percentile >= 0.75 for percentile in percentiles),
    }
    (args.output / "motion_signal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
