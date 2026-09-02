"""Mede se a bola anotada aparece entre as candidatas do detector."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Calcula a sobreposição entre duas caixas xyxy."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia candidate recall da bola por limiar e partida.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation", help="Split a avaliar; use diagnostic para conjuntos de diagnóstico.")
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["split"] == args.split and row["ball_visibility"] in {"VISIBLE", "PARTIAL"}
        ]
    if not rows:
        raise ValueError("Não há positivos VISIBLE/PARTIAL de validação.")
    model = YOLO(str(args.model))
    frame_rows: list[dict[str, str]] = []
    for confidence in args.confidences:
        for index, row in enumerate(rows, start=1):
            image = cv2.imread(row["image_path"])
            if image is None:
                raise FileNotFoundError(row["image_path"])
            result = model.predict(
                source=image,
                conf=confidence,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            gt = tuple(float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"))
            boxes = [tuple(float(value) for value in box) for box in result.boxes.xyxy.tolist()]
            overlaps = [iou(gt, box) for box in boxes]
            matching = [candidate_index for candidate_index, overlap in enumerate(overlaps) if overlap >= args.iou]
            rank = matching[0] + 1 if matching else 0
            frame_rows.append({
                "model_id": args.model_id,
                "confidence": f"{confidence:.2f}",
                "sample_id": row["sample_id"],
                "match_id": row["match_id"],
                "ball_visibility": row["ball_visibility"],
                "candidate_count": str(len(boxes)),
                "best_iou": f"{max(overlaps, default=0.0):.6f}",
                "matching_rank": str(rank),
                "candidate_hit": str(bool(matching)),
                "top3_hit": str(bool(rank and rank <= 3)),
            })
            if index % 25 == 0 or index == len(rows):
                print(f"conf={confidence:.2f} processados={index}/{len(rows)}")

    args.output.mkdir(parents=True, exist_ok=True)
    frame_path = args.output / "candidate_recall_frames.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        groups[(row["confidence"], "all", "all")].append(row)
        groups[(row["confidence"], "match_id", row["match_id"])].append(row)
        groups[(row["confidence"], "ball_visibility", row["ball_visibility"])].append(row)
    summary_rows: list[dict[str, str]] = []
    for (confidence, group_type, group_value), group_rows in sorted(groups.items()):
        total = len(group_rows)
        candidate_hits = sum(row["candidate_hit"] == "True" for row in group_rows)
        top3_hits = sum(row["top3_hit"] == "True" for row in group_rows)
        summary_rows.append({
            "model_id": args.model_id,
            "confidence": confidence,
            "group_type": group_type,
            "group_value": group_value,
            "frames": str(total),
            "candidate_hits": str(candidate_hits),
            "candidate_recall": f"{candidate_hits / total:.6f}",
            "top3_hits": str(top3_hits),
            "top3_recall": f"{top3_hits / total:.6f}",
            "mean_candidate_count": f"{sum(int(row['candidate_count']) for row in group_rows) / total:.3f}",
        })
    summary_path = args.output / "candidate_recall_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"frames={len(rows)}")
    print(f"frame_csv={frame_path}")
    print(f"summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
