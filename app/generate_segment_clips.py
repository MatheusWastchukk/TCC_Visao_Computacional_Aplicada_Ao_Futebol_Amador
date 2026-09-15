"""Gera MP4s de segmentos operacionais usando OpenCV, sem depender de FFmpeg externo."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def clip_bounds(start: float, end: float, duration: float) -> tuple[float, float]:
    """Limita um segmento aos limites conhecidos do vídeo."""
    bounded_start = max(0.0, min(start, duration))
    bounded_end = max(bounded_start, min(end, duration))
    return bounded_start, bounded_end


def render_clip(source: Path, destination: Path, start: float, end: float) -> tuple[float, int]:
    """Renderiza um clipe de vídeo sem áudio e retorna duração e número de quadros."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Metadados inválidos no vídeo de origem.")
        bounded_start, bounded_end = clip_bounds(start, end, frame_count / fps)
        if bounded_end <= bounded_start:
            raise ValueError("O segmento não possui duração dentro do vídeo.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(bounded_start * fps))
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Não foi possível criar o clipe: {destination}")
        written = 0
        try:
            last_frame = int(bounded_end * fps)
            while int(capture.get(cv2.CAP_PROP_POS_FRAMES)) < last_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
        finally:
            writer.release()
        return bounded_end - bounded_start, written
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera clipes MP4 sem áudio para segmentos POSSIBLE_GOAL.")
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    segments = read_csv(args.segments)
    if not segments:
        raise ValueError("Não há segmentos para renderizar.")
    args.output.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    for index, segment in enumerate(segments, start=1):
        name = f"{index:03d}_{segment['segment_id'].lower()}_possible_goal.mp4"
        duration, frames = render_clip(args.video, args.output / name, float(segment["start_seconds"]), float(segment["end_seconds"]))
        manifest.append({"segment_id": segment["segment_id"], "clip_filename": name, "duration_seconds": f"{duration:.3f}", "frames_written": frames, "label": "POSSIBLE_GOAL"})
        print(f"Clipe {index}/{len(segments)}: {name}", flush=True)
    with (args.output / "clips_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    (args.output / "summary.json").write_text(json.dumps({"clips": len(manifest), "audio": "ausente; fallback OpenCV", "warning": "Os clipes são candidatos POSSIBLE_GOAL e exigem revisão humana."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
