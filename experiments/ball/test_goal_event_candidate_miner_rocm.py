"""Valida a inferência ROCm dos dois detectores usados pelo minerador de eventos."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def read_frame(capture: cv2.VideoCapture, fps: float, timestamp: float):
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(timestamp * fps)))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Não foi possível ler o quadro em {timestamp:.3f}s.")
    return frame


def predict_timed(model: YOLO, frame, confidence: float, imgsz: int) -> tuple[int, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = model.predict(source=frame, conf=confidence, imgsz=imgsz, device=0, verbose=False)[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return len(result.boxes), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Testa ROCm com os detectores de estrutura do gol e bola do minerador."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--goal-model", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamps", type=float, nargs="+", default=[60.0, 1200.0, 3000.0])
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.05)
    parser.add_argument("--goal-imgsz", type=int, default=640)
    parser.add_argument("--ball-imgsz", type=int, default=960)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm não disponibilizou a GPU como cuda:0 neste ambiente.")
    for path in (args.video, args.goal_model, args.ball_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps
        frames = [read_frame(capture, fps, timestamp) for timestamp in args.timestamps if timestamp <= duration]
    finally:
        capture.release()
    if not frames:
        raise ValueError("Nenhum timestamp informado pertence ao vídeo.")

    goal_model = YOLO(str(args.goal_model))
    ball_model = YOLO(str(args.ball_model))
    rows = []
    for index, frame in enumerate(frames, start=1):
        goal_count, goal_seconds = predict_timed(goal_model, frame, args.goal_confidence, args.goal_imgsz)
        ball_count, ball_seconds = predict_timed(ball_model, frame, args.ball_confidence, args.ball_imgsz)
        rows.append({
            "frame_index": index,
            "goal_predictions": goal_count,
            "goal_seconds": round(goal_seconds, 6),
            "ball_predictions": ball_count,
            "ball_seconds": round(ball_seconds, 6),
            "combined_seconds": round(goal_seconds + ball_seconds, 6),
        })

    report = {
        "experiment_id": args.output.name,
        "status": "completed",
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "video": str(args.video),
        "frames": len(rows),
        "goal_mean_seconds": round(sum(row["goal_seconds"] for row in rows) / len(rows), 6),
        "ball_mean_seconds": round(sum(row["ball_seconds"] for row in rows) / len(rows), 6),
        "combined_mean_seconds": round(sum(row["combined_seconds"] for row in rows) / len(rows), 6),
        "frame_results": rows,
        "note": "Smoke test de infraestrutura; não é avaliação de eventos nem benchmark completo.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
