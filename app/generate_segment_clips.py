"""Gera clipes H.264 compatíveis com o player do Electron."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
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


def render_clip(source: Path, destination: Path, start: float, end: float, overlay_text: str, evidence_text: str = "") -> tuple[float, int]:
    """Extrai e codifica apenas o intervalo solicitado, com overlay H.264."""
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
    number_match = re.search(r"(\d+)", destination.stem)
    clip_title = f"Clipe {(number_match.group(1) if number_match else '---').zfill(3)}"
    font = "C\\:/Windows/Fonts/arial.ttf"
    filter_graph = (
        "drawbox=x=12:y=12:w=min(iw-24\\,720):h=98:color=0x12362E@1.0:t=fill,"
        f"drawtext=fontfile='{font}':text='{overlay_text}':x=26:y=23:fontsize=24:fontcolor=white,"
        f"drawtext=fontfile='{font}':text='{clip_title}':x=26:y=52:fontsize=16:fontcolor=0xBEEBD6,"
        f"drawtext=fontfile='{font}':text='{evidence_text}':x=26:y=76:fontsize=14:fontcolor=0xBEEBD6"
    )
    command = [
        str(FFMPEG), "-y", "-ss", f"{bounded_start:.3f}", "-to", f"{bounded_end:.3f}", "-i", str(source),
        "-vf", filter_graph, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg não conseguiu codificar o clipe: {completed.stderr[-500:]}")
    return bounded_end - bounded_start, round((bounded_end - bounded_start) * fps)


def main(argv: list[str] | None = None, progress: Callable[[str, int], None] | None = None) -> int:
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
        segment_match = re.search(r"(\d+)$", segment["segment_id"])
        number = (segment_match.group(1) if segment_match else str(index)).zfill(3)
        name = f"clipe_{number}_possible_goal.mp4"
        evidence = f"Rank: {segment.get('rank_score', '—')} | Score: {segment.get('fusion_score_goal', '—')}"
        duration, written = render_clip(args.video, args.output / name, float(segment["start_seconds"]), float(segment["end_seconds"]), "POSSIBLE_GOAL", evidence)
        manifest.append({"segment_id": segment["segment_id"], "clip_filename": name, "duration_seconds": f"{duration:.3f}", "frames_written": written, "label": "POSSIBLE_GOAL"})
        print(f"Clipe {index}/{len(segments)}: {name}", flush=True)
        if progress:
            progress(f"Gerando clipe {index}/{len(segments)}", 85 + round(15 * index / len(segments)))
    with (args.output / "clips_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    (args.output / "summary.json").write_text(json.dumps({"clips": len(manifest), "audio": "ausente", "codec": "H.264"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
