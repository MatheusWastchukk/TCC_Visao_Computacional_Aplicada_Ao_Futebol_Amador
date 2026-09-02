"""Prepara candidatas do detector para treinar um verificador visual da bola."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara candidatas para o verificador visual da bola.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequence_rows = list(csv.DictReader(file))
    model = YOLO(str(args.model))
    output_rows: list[dict[str, str]] = []
    for frame_index, row in enumerate(sequence_rows, start=1):
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        result = model.predict(source=image, conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        candidates = sorted(
            [(tuple(float(value) for value in box), float(confidence)) for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())],
            key=lambda item: item[1], reverse=True,
        )[: args.max_candidates]
        target = tuple(json.loads(row["ball_of_play_box_json"])) if row["ball_of_play_box_json"] else None
        distractors = [tuple(values) for values in json.loads(row["distractor_boxes_json"] or "[]")]
        for candidate_rank, (box, confidence) in enumerate(candidates, start=1):
            target_iou = iou(box, target) if target else 0.0
            distractor_iou = max((iou(box, distractor) for distractor in distractors), default=0.0)
            if target_iou >= 0.5:
                label, status, source = "BOLA_JOGO", "completed", "anotacao_alvo"
            elif distractor_iou >= 0.5:
                label, status, source = "BOLA_DISTRATORA", "completed", "anotacao_distrator"
            else:
                label, status, source = "", "pending", "revisao_manual"
            output_rows.append({
                "candidate_id": f"verifier_v1_{frame_index:03d}_{candidate_rank:02d}",
                "sequence_id": row["sequence_id"], "sequence_frame": row["sequence_frame"],
                "sample_id": row["sample_id"], "match_id": row["match_id"],
                "image_path": row["image_path"], "candidate_rank": str(candidate_rank),
                "confidence": f"{confidence:.6f}", "box_json": json.dumps(box),
                "target_iou": f"{target_iou:.6f}", "distractor_iou": f"{distractor_iou:.6f}",
                "label": label, "annotation_status": status, "label_source": source, "notes": "",
            })
        print(f"[{frame_index}/{len(sequence_rows)}] {row['sample_id']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    completed = sum(row["annotation_status"] == "completed" for row in output_rows)
    print(f"candidatas={len(output_rows)} pre_rotuladas={completed} pendentes={len(output_rows) - completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
