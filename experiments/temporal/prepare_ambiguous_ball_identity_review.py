"""Prepara frames ambíguos para identificar a bola efetiva da jogada."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrai detecções de bola nos casos de associação ambígua para revisão manual."
    )
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    reviewed = [
        row
        for row in read_csv(args.review_manifest)
        if row.get("review_label") == "ASSOCIACAO_INCORRETA"
    ]
    if not reviewed:
        raise ValueError("Não há sequências marcadas como ASSOCIACAO_INCORRETA.")
    relations = {
        (row["block_id"], row["timestamp_hhmmss"]): row for row in read_csv(args.relations)
    }
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True)
    captures: dict[str, cv2.VideoCapture] = {}
    output_rows: list[dict[str, str]] = []
    try:
        for index, sequence in enumerate(reviewed, start=1):
            video_name = sequence["video_filename"]
            capture = captures.get(video_name)
            if capture is None:
                capture = cv2.VideoCapture(str(args.videos_dir / video_name))
                if not capture.isOpened():
                    raise FileNotFoundError(args.videos_dir / video_name)
                captures[video_name] = capture
            timestamp = sequence["representative_timestamp_hhmmss"]
            relation = relations.get((sequence["block_id"], timestamp))
            if relation is None:
                raise KeyError(f"Relação não encontrada para {sequence['sequence_id']} em {timestamp}.")
            capture.set(cv2.CAP_PROP_POS_MSEC, float(relation["timestamp_seconds"]) * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Não foi possível ler {video_name} em {timestamp}.")
            result = model.predict(
                source=image,
                imgsz=args.imgsz,
                conf=args.confidence,
                device=args.device,
                verbose=False,
            )[0]
            boxes = [[round(float(value), 2) for value in box] for box in result.boxes.xyxy.tolist()]
            confidences = [round(float(value), 6) for value in result.boxes.conf.tolist()]
            frame_path = frames_dir / f"{index:03d}_{sequence['sequence_id']}.jpg"
            if not cv2.imwrite(str(frame_path), image):
                raise RuntimeError(f"Não foi possível salvar {frame_path}.")
            output_rows.append({
                "ambiguous_id": f"ambiguous_ball_{index:03d}",
                "sequence_id": sequence["sequence_id"],
                "block_id": sequence["block_id"],
                "match_id": sequence["match_id"],
                "video_filename": video_name,
                "timestamp_hhmmss": timestamp,
                "frames_inside_goal_box": sequence["frames_inside_goal_box"],
                "frame_path": frame_path.as_posix(),
                "candidate_boxes_json": json.dumps(boxes),
                "candidate_confidences_json": json.dumps(confidences),
                "previous_track_box_json": json.dumps([
                    relation["ball_x1"], relation["ball_y1"], relation["ball_x2"], relation["ball_y2"]
                ]),
                "annotation_status": "pending",
                "target_candidate_id": "",
                "target_visibility": "",
                "annotation_notes": "",
            })
            print(f"Preparado: {index}/{len(reviewed)}")
    finally:
        for capture in captures.values():
            capture.release()

    target = args.output / "ambiguous_ball_identity_manifest.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Manifesto: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
