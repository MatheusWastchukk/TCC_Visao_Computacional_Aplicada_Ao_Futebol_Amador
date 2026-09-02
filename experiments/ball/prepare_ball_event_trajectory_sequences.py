"""Extrai sequências ao redor de momentos positivos para avaliar trajetória e entrada no gol."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from prepare_ball_identity_sequences import format_time


FIELDS = [
    "sequence_id", "sequence_frame", "sample_id", "event_id", "is_event_frame", "match_id", "video_filename",
    "capture_domain", "split", "timestamp_seconds", "timestamp_hhmmss", "image_path", "ball_visibility",
    "ball_of_play_box_json", "distractor_boxes_json", "goal_region_box_json", "annotation_status", "annotator", "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara sequências ao redor de instantes anotados de eventos.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--moments", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-before", type=float, default=2.0)
    parser.add_argument("--seconds-after", type=float, default=2.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument(
        "--event-labels", nargs="+", default=["POSSIVEL_GOL"],
        help="Rótulos de evento a extrair; o padrão preserva somente POSSIVEL_GOL.",
    )
    args = parser.parse_args()
    with args.events.open(newline="", encoding="utf-8") as file:
        events = {
            row["event_id"]: row for row in csv.DictReader(file)
            if row["event_label"] in set(args.event_labels)
        }
    with args.moments.open(newline="", encoding="utf-8") as file:
        moments = [row for row in csv.DictReader(file) if row["annotation_status"] == "completed" and row.get("event_timestamp_seconds")]
    step = 1.0 / args.sample_fps
    offsets = []
    value = -args.seconds_before
    while value <= args.seconds_after + 1e-9:
        offsets.append(value)
        value += step
    output_rows: list[dict[str, str]] = []
    for sequence_number, moment in enumerate(moments, start=1):
        event = events.get(moment["event_id"])
        if event is None:
            raise KeyError(f"Evento positivo ausente para {moment['event_id']}.")
        video_path = args.videos_dir / event["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            event_seconds = float(moment["event_timestamp_seconds"])
            sequence_id = f"event_trajectory_sequence_{sequence_number:02d}"
            event_frame_number = min(range(1, len(offsets) + 1), key=lambda index: abs(offsets[index - 1]))
            for frame_number, offset in enumerate(offsets, start=1):
                seconds = min(max(0.0, event_seconds + offset), (frame_count - 1) / fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Não foi possível ler {video_path} em {seconds:.3f} s.")
                sample_id = f"{sequence_id}_frame_{frame_number:03d}"
                image_path = args.frames_dir / event["match_id"] / f"{sample_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Não foi possível salvar {image_path}.")
                output_rows.append({
                    "sequence_id": sequence_id, "sequence_frame": str(frame_number), "sample_id": sample_id,
                    "event_id": event["event_id"], "is_event_frame": str(frame_number == event_frame_number),
                    "match_id": event["match_id"], "video_filename": event["video_filename"], "capture_domain": "Veo/derivados",
                    "split": "development", "timestamp_seconds": f"{seconds:.3f}", "timestamp_hhmmss": format_time(seconds),
                    "image_path": image_path.as_posix(), "ball_visibility": "VISIBLE", "ball_of_play_box_json": "",
                    "distractor_boxes_json": "[]", "goal_region_box_json": "", "annotation_status": "pending", "annotator": "",
                    "notes": f"origem={event['event_id']}; rotulo={event['event_label']}; delta_seconds={offset:+.1f}; uso=avaliacao_evento",
                })
        finally:
            capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"sequências={len(moments)}")
    print(f"frames={len(output_rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
