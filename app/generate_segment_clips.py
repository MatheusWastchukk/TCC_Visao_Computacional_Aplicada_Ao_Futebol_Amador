"""Gera clipes H.264 compatíveis com o player do Electron."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / "desktop" / "node_modules" / "ffmpeg-static" / "ffmpeg.exe"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def clip_bounds(start: float, end: float, duration: float) -> tuple[float, float]:
    bounded_start = max(0.0, min(start, duration))
    bounded_end = max(bounded_start, min(end, duration))
    return bounded_start, bounded_end


def render_clip(
    source: Path, destination: Path, start: float, end: float,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[float, int]:
    """Extrai e codifica o intervalo solicitado sem informações sobrepostas."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or frames <= 0:
            raise ValueError("Metadados inválidos no vídeo de origem.")
        bounded_start, bounded_end = clip_bounds(start, end, frames / fps)
    finally:
        capture.release()
    if bounded_end <= bounded_start:
        raise ValueError("O clipe não possui duração dentro do vídeo.")
    if not FFMPEG.is_file():
        raise RuntimeError("FFmpeg local não encontrado. Execute npm install na pasta desktop.")
    if cancel_check:
        cancel_check()
    command = [
        str(FFMPEG), "-v", "error", "-y", "-ss", f"{bounded_start:.3f}", "-to", f"{bounded_end:.3f}", "-i", str(source),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as error_log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error_log)
        try:
            while process.poll() is None:
                if cancel_check:
                    cancel_check()
                time.sleep(0.1)
            error_log.seek(0)
            stderr = error_log.read()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            destination.unlink(missing_ok=True)
            raise
    completed = subprocess.CompletedProcess(command, process.returncode, None, stderr)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg não conseguiu codificar o clipe: {completed.stderr[-500:]}")
    return bounded_end - bounded_start, round((bounded_end - bounded_start) * fps)


def main(
    argv: list[str] | None = None,
    progress: Callable[[str, int], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Gera MP4s H.264 para segmentos POSSIBLE_GOAL.")
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
        if cancel_check:
            cancel_check()
        segment_match = re.search(r"(\d+)$", segment["segment_id"])
        number = (segment_match.group(1) if segment_match else str(index)).zfill(3)
        name = f"clipe_{number}_possible_goal.mp4"
        duration, written = render_clip(
            args.video, args.output / name, float(segment["start_seconds"]), float(segment["end_seconds"]), cancel_check,
        )
        manifest.append({"segment_id": segment["segment_id"], "clip_filename": name, "duration_seconds": f"{duration:.3f}", "frames_written": written, "label": "POSSIBLE_GOAL", "video_overlay": "none"})
        print(f"Clipe {index}/{len(segments)}: {name}", flush=True)
        if progress:
            progress(f"Gerando clipe {index}/{len(segments)}", 85 + round(15 * index / len(segments)))
    if cancel_check:
        cancel_check()
    with (args.output / "clips_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    (args.output / "summary.json").write_text(json.dumps({"clips": len(manifest), "audio": "ausente", "codec": "H.264"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
