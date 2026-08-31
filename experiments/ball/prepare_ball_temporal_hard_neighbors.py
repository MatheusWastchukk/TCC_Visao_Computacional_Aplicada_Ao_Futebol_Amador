"""Extrai vizinhos temporais de positivos difíceis já pertencentes ao treino."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from prepare_ball_dataset_candidates import FIELDS, extract_frame, format_time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara vizinhos temporais de exemplos difíceis do split de treino da bola."
    )
    parser.add_argument("--hard-examples", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=10)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.4, 0.4])
    args = parser.parse_args()

    rows = list(csv.DictReader(args.hard_examples.open(newline="", encoding="utf-8")))
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            by_match[row["match_id"]].append(row)
    ranked_matches = sorted(by_match, key=lambda match_id: (-len(by_match[match_id]), match_id))
    selected = [
        min(by_match[match_id], key=lambda row: (float(row["mined_iou"]), row["sample_id"]))
        for match_id in ranked_matches[: args.contexts]
    ]
    if not selected:
        raise ValueError("Não foram encontrados exemplos difíceis de treino para extrair vizinhos.")

    output_rows: list[dict[str, str]] = []
    for context_index, center in enumerate(selected, start=1):
        video_path = args.videos_dir / center["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        center_time = float(center["timestamp_seconds"])
        for offset_index, offset in enumerate(args.offsets, start=1):
            timestamp = min(max(0.0, center_time + offset), (frame_count - 1) / fps)
            frame_index = round(timestamp * fps)
            sample_id = f"{center['match_id']}_hard_temporal_{context_index:02d}_{offset_index:02d}"
            image_path = args.frames_dir / center["match_id"] / f"{sample_id}_{timestamp:010.3f}.jpg"
            extract_frame(video_path, frame_index, image_path)
            output_rows.append({
                "sample_id": sample_id,
                "match_id": center["match_id"],
                "video_filename": center["video_filename"],
                "capture_domain": center["capture_domain"],
                "split": "train",
                "timestamp_seconds": f"{timestamp:.3f}",
                "timestamp_hhmmss": format_time(timestamp),
                "image_path": image_path.as_posix(),
                "ball_visibility": "pending",
                "ball_x1": "",
                "ball_y1": "",
                "ball_x2": "",
                "ball_y2": "",
                "annotation_status": "pending",
                "annotator": "",
                "notes": (
                    f"origem=vizinho_temporal_hard; centro={center['sample_id']}; "
                    f"delta_seconds={offset:+.1f}; frame_index={frame_index}"
                ),
            })
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"contextos_treino={len(selected)}")
    print(f"frames_para_anotar={len(output_rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
