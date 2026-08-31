"""Avalia um detector treinado em tiles no frame original completo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


Box = tuple[float, float, float, float]


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def nms(boxes: list[tuple[Box, float]], threshold: float) -> list[tuple[Box, float]]:
    """Aplica NMS simples para uma única classe, em coordenadas globais."""
    ordered = sorted(boxes, key=lambda item: item[1], reverse=True)
    kept: list[tuple[Box, float]] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [item for item in ordered if iou(current[0], item[0]) < threshold]
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia um modelo de tiles nos frames originais.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def tile_bounds(width: int, height: int) -> list[tuple[int, int, int, int]]:
    mid_x, mid_y = width // 2, height // 2
    return [(0, 0, mid_x, mid_y), (mid_x, 0, width, mid_y), (0, mid_y, mid_x, height), (mid_x, mid_y, width, height)]


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "validation"
            and row["ball_visibility"] in {"VISIBLE", "PARTIAL", "NOT_VISIBLE"}
        ]

    model = YOLO(str(args.model))
    audit_rows: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")
        height, width = image.shape[:2]
        candidates: list[tuple[Box, float]] = []
        for left, top, right, bottom in tile_bounds(width, height):
            tile = image[top:bottom, left:right]
            result = model.predict(
                source=tile,
                conf=args.conf,
                iou=0.7,
                imgsz=args.imgsz,
                verbose=False,
                device=args.device,
            )[0]
            for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                candidates.append(((x1 + left, y1 + top, x2 + left, y2 + top), float(confidence)))

        kept = nms(candidates, args.nms_iou)
        best_box, best_conf = kept[0] if kept else (None, 0.0)
        has_ground_truth = row["ball_visibility"] in {"VISIBLE", "PARTIAL"}
        ground_truth = tuple(float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")) if has_ground_truth else None
        overlap = iou(ground_truth, best_box) if ground_truth and best_box else 0.0
        preliminary = ("TP" if best_box and overlap >= args.match_iou else "FN") if has_ground_truth else ("TN" if best_box is None else "FP")

        overlay = image.copy()
        if ground_truth:
            x1, y1, x2, y2 = map(int, ground_truth)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(overlay, "GT", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        if best_box:
            x1, y1, x2, y2 = map(int, best_box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(overlay, f"P {best_conf:.2f}", (x1, min(height - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(overlay, f"{preliminary} | {row['ball_visibility']} | tiles=4", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        overlay_path = args.output / "overlays" / f"{row['sample_id']}.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)
        audit_rows.append({
            "sample_id": row["sample_id"],
            "match_id": row["match_id"],
            "capture_domain": row["capture_domain"],
            "ball_visibility": row["ball_visibility"],
            "timestamp_hhmmss": row["timestamp_hhmmss"],
            "preliminary_status": preliminary,
            "prediction_confidence": f"{best_conf:.6f}" if best_box else "",
            "iou": f"{overlap:.6f}",
            "overlay_path": overlay_path.as_posix(),
            "review_status": "pending",
            "review_label": "",
            "review_notes": "",
        })
        if number % 20 == 0 or number == len(rows):
            print(f"Processados: {number}/{len(rows)}")

    csv_path = args.output / "ball_prediction_audit_tiles_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    counts: dict[str, int] = {}
    for row in audit_rows:
        counts[row["preliminary_status"]] = counts.get(row["preliminary_status"], 0) + 1
    print(f"frames={len(audit_rows)}")
    print(f"status={counts}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
