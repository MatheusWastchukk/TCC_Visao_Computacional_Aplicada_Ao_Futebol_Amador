"""Renderiza overlays do detector nos frames dos falsos negativos temporais."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Renderiza as caixas previstas nos falsos negativos temporais.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def draw_text(image, text: str, origin: tuple[int, int], color=(255, 255, 255)) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)


def main() -> int:
    args = parse_args()
    rows = read_csv(args.review_csv)
    if not rows:
        raise ValueError("Nenhum frame de revisão encontrado.")
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    rendered_rows: list[dict[str, str]] = []

    for row in rows:
        video_path = args.videos_dir / row["video_filename"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(row["timestamp_seconds"]) * 1000.0)
        ok, image = cap.read()
        cap.release()
        if not ok or image is None:
            continue

        result = model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.confidence,
            device=args.device,
            verbose=False,
        )[0]
        rendered = image.copy()
        boxes = result.boxes
        prediction_count = len(boxes) if boxes is not None else 0
        max_confidence = 0.0
        if boxes is not None and len(boxes):
            coordinates = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            max_confidence = float(confidences.max())
            for box, confidence in zip(coordinates, confidences, strict=True):
                x1, y1, x2, y2 = [int(value) for value in box]
                cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 220, 0), 3)
                cv2.putText(
                    rendered,
                    f"goal_structure {confidence:.3f}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 0),
                    2,
                )

        draw_text(rendered, f"GT humano: goal_structure | Resultado: false_negative", (15, 30), (0, 255, 255))
        draw_text(rendered, f"{row['video_filename']} | {row['timestamp_hhmmss']}", (15, 60))
        prediction_text = f"Predicoes: {prediction_count} | Maior confianca: {max_confidence:.3f}"
        prediction_color = (0, 220, 0) if prediction_count else (0, 0, 255)
        draw_text(rendered, prediction_text, (15, 90), prediction_color)
        filename = Path(row["frame_path"]).name.replace(".jpg", "_overlay.jpg")
        output_path = args.output / filename
        cv2.imwrite(str(output_path), rendered)
        rendered_rows.append(
            {
                **row,
                "overlay_path": str(output_path),
                "overlay_prediction_count": str(prediction_count),
                "overlay_max_confidence": f"{max_confidence:.6f}",
            }
        )

    output_csv = args.output / "false_negative_overlays_v1.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rendered_rows[0]))
        writer.writeheader()
        writer.writerows(rendered_rows)
    print(f"overlays={len(rendered_rows)}")
    print(f"csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
