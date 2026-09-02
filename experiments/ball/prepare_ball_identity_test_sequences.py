"""Extrai sequências temporais de partidas fechadas para testar a identidade da bola."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


FIELDS = [
    "sequence_id", "sequence_frame", "sample_id", "match_id", "video_filename", "capture_domain",
    "split", "timestamp_seconds", "timestamp_hhmmss", "image_path", "ball_visibility",
    "ball_of_play_box_json", "distractor_boxes_json", "annotation_status", "annotator", "notes",
]


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def spaced(items: list[str], count: int) -> list[str]:
    """Escolhe itens distribuídos pela lista, sem repetir partidas."""
    if count >= len(items):
        return items
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)] if count > 1 else [items[len(items) // 2]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara sequências fechadas para testar a identidade temporal da bola.")
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=6)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.4, -0.2, 0.0, 0.2, 0.4])
    args = parser.parse_args()

    with args.frame_manifest.open(newline="", encoding="utf-8") as file:
        frame_rows = list(csv.DictReader(file))
    with args.split_manifest.open(newline="", encoding="utf-8") as file:
        metadata = {row["match_id"]: row for row in csv.DictReader(file) if row["split"] == "test"}
    contexts_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        if row["match_id"] in metadata and row["goal_structure_label"] == "goal_structure":
            contexts_by_match[row["match_id"]].append(row)
    selected_matches = spaced(sorted(metadata), args.contexts)
    if not selected_matches:
        raise ValueError("Não há partidas disponíveis no split test.")

    output_rows: list[dict[str, str]] = []
    for sequence_number, match_id in enumerate(selected_matches, start=1):
        contexts = sorted(contexts_by_match[match_id], key=lambda row: float(row["timestamp"]))
        center = contexts[len(contexts) // 2] if contexts else None
        details = metadata[match_id]
        video_path = args.videos_dir / details["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            sequence_id = f"identity_test_sequence_{sequence_number:02d}"
            duration_seconds = (frame_count - 1) / fps
            center_seconds = float(center["timestamp"]) if center else duration_seconds * (0.20 + 0.12 * ((sequence_number - 1) % 6))
            for frame_number, offset in enumerate(args.offsets, start=1):
                seconds = min(max(0.0, center_seconds + offset), (frame_count - 1) / fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Não foi possível ler {video_path} em {seconds:.3f} s.")
                sample_id = f"{sequence_id}_frame_{frame_number:02d}"
                image_path = args.frames_dir / match_id / f"{sample_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Não foi possível salvar {image_path}.")
                output_rows.append({
                    "sequence_id": sequence_id, "sequence_frame": str(frame_number), "sample_id": sample_id,
                    "match_id": match_id, "video_filename": details["video_filename"], "capture_domain": details["capture_domain"],
                    "split": "test", "timestamp_seconds": f"{seconds:.3f}", "timestamp_hhmmss": format_time(seconds),
                    "image_path": image_path.as_posix(), "ball_visibility": "VISIBLE", "ball_of_play_box_json": "",
                    "distractor_boxes_json": "[]", "annotation_status": "pending", "annotator": "",
                    "notes": (
                        f"origem={center['frame_id'] if center else 'amostragem_temporal_neutra'}; "
                        f"delta_seconds={offset:+.1f}; uso=teste_fechado_identidade_temporal"
                    ),
                })
        finally:
            capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"sequências={len(selected_matches)}")
    print(f"frames={len(output_rows)}")
    print(f"partidas={','.join(selected_matches)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
