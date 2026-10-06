"""Gera arquivos MP4 completos para a fila da auditoria de ranking TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
DEFAULT_FFMPEG = ROOT / "desktop/node_modules/ffmpeg-static/ffmpeg.exe"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def encode(ffmpeg: Path, source: Path, destination: Path, start: float, duration: float,
           gpu: bool) -> tuple[bool, str]:
    base = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
            "-i", str(source), "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
    if gpu:
        command = base + ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp",
                          "-qp_i", "24", "-qp_p", "26", str(destination)]
    else:
        command = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0, result.stderr[-1200:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    args = parser.parse_args()
    source_manifest = args.experiment / "manual_case_audit.csv"
    output_dir = args.experiment / "full_clips_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(f"FFmpeg local não encontrado: {args.ffmpeg}")
    cases = read_rows(source_manifest)
    progress_path = output_dir / "progress.json"
    output_manifest = output_dir / "full_clips_manifest.csv"
    existing = {row["case_id"]: row for row in read_rows(output_manifest)} if output_manifest.is_file() else {}
    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        previous = existing.get(case_id, {})
        destination = output_dir / previous.get("filename", f"case_{index:03d}_{case['match_id']}_{case['group_uid'].split('::')[-1]}.mp4")
        if destination.is_file() and previous.get("status") == "COMPLETE":
            print(f"{index}/{len(cases)} já pronto: {destination.name}", flush=True)
            continue
        source = args.videos_dir / case["video_filename"]
        if not source.is_file():
            raise FileNotFoundError(f"Vídeo original não encontrado: {source}")
        start = float(case["delivery_start_seconds"])
        end = float(case["delivery_end_seconds"])
        duration = end - start
        if duration <= 0:
            raise ValueError(f"Intervalo inválido no caso {case_id}: {start}–{end}")
        ok, detail = encode(args.ffmpeg, source, destination, start, duration, gpu=True)
        encoder = "AMD_AMF"
        if not ok:
            destination.unlink(missing_ok=True)
            print(f"GPU AMD falhou no caso {case_id}; usando CPU. {detail}", flush=True)
            ok, detail = encode(args.ffmpeg, source, destination, start, duration, gpu=False)
            encoder = "libx264_CPU_fallback"
        if not ok:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Falha de codificação no caso {case_id}: {detail}")
        existing[case_id] = {
            "case_id": case_id, "match_id": case["match_id"], "video_filename": case["video_filename"],
            "filename": destination.name, "candidate_timestamp_seconds": case["best_timestamp_seconds"],
            "start_seconds": f"{start:.3f}", "end_seconds": f"{end:.3f}",
            "duration_seconds": f"{duration:.3f}", "temporal_rank_full_pool": case["temporal_rank_full_pool"],
            "top60_rank": case["top60_rank"], "review_role": case["review_role"],
            "encoder": encoder, "overlays": "none; complete original footage for human review", "status": "COMPLETE",
        }
        ordered = [existing[row["case_id"]] for row in cases if row["case_id"] in existing]
        write_rows(output_manifest, ordered)
        progress_path.write_text(json.dumps({"completed": len(ordered), "total": len(cases),
                                             "last_case_id": case_id}, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"{index}/{len(cases)}: {destination.name} ({duration:.1f}s, {encoder})", flush=True)
    readme = output_dir / "README.txt"
    readme.write_text(
        "175 clipes completos da auditoria TRAIN. Cada MP4 cobre o intervalo da fila: 20 s antes e 40 s depois do candidato, limitado ao vídeo original.\n"
        "Estes arquivos não têm caixas de detecção, para não apresentar propostas parciais/esparsas como overlay contínuo.\n"
        "O CSV full_clips_manifest.csv liga cada arquivo ao caso, partida, rank e timestamp original.\n"
        "As caixas históricas disponíveis são propostas em frames amostrados, não tracking. Use o marcador OpenCV para inspecioná-las; ele mostra todas as propostas persistidas no frame amostrado.\n",
        encoding="utf-8")
    print(f"Concluído: {len(cases)} clipes em {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
