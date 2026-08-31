"""Gera overlays e uma tabela para auditoria visual do detector da bola."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera overlays para auditar o detector da bola.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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
    for row in rows:
        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")
        result = model.predict(source=image, conf=args.conf, iou=0.7, imgsz=args.imgsz, verbose=False, device=args.device)[0]
        predictions = result.boxes
        best_index = -1
        best_conf = 0.0
        best_box: tuple[float, float, float, float] | None = None
        for index, (box, confidence) in enumerate(zip(predictions.xyxy.tolist(), predictions.conf.tolist())):
            if confidence > best_conf:
                best_index = index
                best_conf = float(confidence)
                best_box = tuple(float(value) for value in box)

        has_ground_truth = row["ball_visibility"] in {"VISIBLE", "PARTIAL"}
        ground_truth = tuple(float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")) if has_ground_truth else None
        overlap = iou(ground_truth, best_box) if ground_truth and best_box else 0.0
        if has_ground_truth:
            preliminary = "TP" if best_box and overlap >= args.iou else "FN"
        else:
            preliminary = "TN" if best_box is None else "FP"

        overlay = image.copy()
        if ground_truth:
            x1, y1, x2, y2 = map(int, ground_truth)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(overlay, "GT", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        if best_box:
            x1, y1, x2, y2 = map(int, best_box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(overlay, f"P {best_conf:.2f}", (x1, min(image.shape[0] - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(overlay, f"{preliminary} | {row['ball_visibility']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
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

    csv_path = args.output / "ball_prediction_audit_v1.csv"
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
