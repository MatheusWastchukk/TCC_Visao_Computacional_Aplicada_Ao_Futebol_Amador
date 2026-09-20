"""API local para o aplicativo Electron de análise de lances."""
from __future__ import annotations

import csv
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import psutil
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.analyze_match import main as analyze_main
from app.generate_segment_clips import main as clips_main
from app.generate_segment_clips import FFMPEG, render_clip
from app.review_storage import load_annotation_rows, read_csv, write_annotation_rows

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Goal Event Review API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
sessions: dict[str, dict[str, object]] = {}
annotation_lock = threading.Lock()


class StartRequest(BaseModel):
    video_path: str
    mode: str = "alta_cobertura"


class TagRequest(BaseModel):
    tag: str
    notes: str = ""


class CompilationRequest(BaseModel):
    segment_ids: list[str]


def session_or_404(session_id: str) -> dict[str, object]:
    if session_id not in sessions:
        raise HTTPException(404, "Sessão não encontrada.")
    return sessions[session_id]


def clip_number(segment_id: str) -> str:
    match = re.search(r"(\d+)$", segment_id)
    return (match.group(1) if match else segment_id).zfill(3)


def label_name(label: str | None) -> str:
    return (label or "POSSIBLE_GOAL").lower()


def output_name(segment_id: str, label: str | None) -> str:
    return f"clipe_{clip_number(segment_id)}_{label_name(label)}.mp4"


def device_metrics() -> dict[str, object]:
    process = psutil.Process(os.getpid())
    metrics: dict[str, object] = {
        "processor": "GPU" if torch.cuda.is_available() else "CPU",
        "process_cpu_percent": round(process.cpu_percent(interval=None), 1),
        "process_ram_mb": round(process.memory_info().rss / 1024**2, 1),
        "process_ram_percent": round(process.memory_percent(), 2),
        "cpu_name": platform.processor() or platform.uname().processor or "CPU do sistema",
    }
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        used_bytes = total_bytes - free_bytes
        metrics.update({
            "device_name": torch.cuda.get_device_name(0),
            "gpu_memory_used_mb": round(used_bytes / 1024**2, 1),
            "gpu_memory_total_mb": round(total_bytes / 1024**2, 1),
            "gpu_memory_percent": round(used_bytes / total_bytes * 100, 1),
            "gpu_process_memory_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
        })
    return metrics


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_manifest_filename(output: Path, segment_id: str, filename: str) -> None:
    path = output / "clips" / "clips_manifest.csv"
    rows = read_csv(path)
    for row in rows:
        if row["segment_id"] == segment_id:
            row["clip_filename"] = filename
            write_rows(path, rows)
            return
    raise HTTPException(404, "Clipe não encontrado.")


def update_manifest_label(output: Path, segment_id: str, label: str) -> None:
    path = output / "clips" / "clips_manifest.csv"
    rows = read_csv(path)
    for row in rows:
        if row["segment_id"] == segment_id:
            row["label"] = label
            write_rows(path, rows)
            return
    raise HTTPException(404, "Clipe não encontrado.")


def session_rows(state: dict[str, object]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    output = Path(state["output"])
    rows, _ = load_annotation_rows(output / "clips" / "clips_manifest.csv", output / "clips" / "clip_annotations.csv")
    segments = {row["segment_id"]: row for row in read_csv(output / "segments.csv")}
    return rows, segments


def evidence_text(segment: dict[str, str]) -> str:
    return f"Rank: {segment.get('rank_score', '—')} | Score: {segment.get('fusion_score_goal', '—')}"


def ensure_export_overlay(state: dict[str, object], row: dict[str, str]) -> Path:
    """Renderiza a tag persistida somente na exportação, sem travar o salvamento."""
    output = Path(state["output"])
    label = row.get("human_label") or (row.get("label") if row.get("label") != "POSSIBLE_GOAL" else "")
    current = output / "clips" / row["clip_filename"]
    if not label:
        return current
    target_name = output_name(row["segment_id"], label)
    target = output / "clips" / target_name
    segment = next(item for item in read_csv(output / "segments.csv") if item["segment_id"] == row["segment_id"])
    render_clip(Path(state["video"]), target, float(segment["start_seconds"]), float(segment["end_seconds"]), label, evidence_text(segment))
    if row["clip_filename"] != target_name:
        update_manifest_filename(output, row["segment_id"], target_name)
    return target


def compile_clips(paths: list[Path], destination: Path) -> None:
    if not FFMPEG.is_file():
        raise RuntimeError("FFmpeg local não encontrado.")
    list_path = destination.with_suffix(".txt")
    entries = ["file '{}'\n".format(path.as_posix().replace("'", "'\\\\''")) for path in paths]
    list_path.write_text("".join(entries), encoding="utf-8")
    try:
        completed = subprocess.run([
            str(FFMPEG), "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ], capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"Não foi possível criar a compilação: {completed.stderr[-500:]}")
    finally:
        list_path.unlink(missing_ok=True)


def run_session(session_id: str, request: StartRequest) -> None:
    state = session_or_404(session_id)
    video = Path(request.video_path)
    output = Path(state["output"])
    started = time.perf_counter()
    try:
        match_id = re.sub(r"[^a-zA-Z0-9_]+", "_", video.stem).strip("_") or "video"

        def progress(label: str, value: float) -> None:
            elapsed = time.perf_counter() - started
            percentage = max(0.0, min(float(value), 100.0))
            estimate = elapsed / percentage * 100.0 if percentage >= 1.0 else None
            state.update({"stage": label, "progress": percentage, "elapsed": elapsed, "estimated_total": estimate, "metrics": device_metrics()})
            print(f"[{time.strftime('%H:%M:%S')}] {label} ({percentage:.2f}%)", flush=True)

        models = ROOT / "experiments/ball"
        analyze_main([
            "--video-path", str(video), "--match-id", match_id, "--mode", request.mode,
            "--output", str(output), "--videos-dir", str(video.parent), "--device", "cuda:0",
            "--allow-arbitrary-video", "--split-manifest", str(ROOT / "experiments/datasets/dataset_split_v1.csv"),
            "--goal-model", str(ROOT / "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt"),
            "--ball-model", str(ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"),
            "--temporal-ranker", str(models / "GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib"),
            "--visual-verifier", str(models / "GOAL-EVENT-PIPELINE-FINAL-V1/temporal_sequence_verifier.joblib"),
            "--fusion-model", str(models / "GOAL-EVENT-FUSION-FINAL-V1/fusion_reranker.joblib"),
        ], progress)
        clips_main(["--segments", str(output / "segments.csv"), "--video", str(video), "--output", str(output / "clips")], progress)
        print(f"[{time.strftime('%H:%M:%S')}] Análise concluída.", flush=True)
        state.update({"stage": "Concluído", "progress": 100.0, "status": "completed", "elapsed": time.perf_counter() - started, "metrics": device_metrics()})
    except Exception as error:
        print(f"[{time.strftime('%H:%M:%S')}] ERRO: {error}", flush=True)
        state.update({"status": "failed", "error": str(error), "elapsed": time.perf_counter() - started, "metrics": device_metrics()})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.3.7"}


@app.post("/sessions")
def start(request: StartRequest) -> dict[str, str]:
    if not Path(request.video_path).is_file():
        raise HTTPException(400, "Vídeo não encontrado.")
    session_id = uuid.uuid4().hex
    workspace = Path(tempfile.mkdtemp(prefix="goal_event_api_"))
    sessions[session_id] = {
        "workspace": str(workspace), "output": str(workspace / "analysis"), "video": request.video_path,
        "status": "running", "stage": "Iniciando análise", "progress": 0.0, "elapsed": 0.0,
        "estimated_total": None, "metrics": device_metrics(), "renders": {},
    }
    threading.Thread(target=run_session, args=(session_id, request), daemon=True).start()
    return {"session_id": session_id}


@app.get("/sessions/{session_id}")
def status(session_id: str) -> dict[str, object]:
    state = session_or_404(session_id)
    state["metrics"] = device_metrics()
    return {key: value for key, value in state.items() if key not in {"output", "workspace", "video"}}


@app.get("/sessions/{session_id}/clips")
def clips(session_id: str) -> list[dict[str, str]]:
    state = session_or_404(session_id)
    rows, segments = session_rows(state)
    renders = state.get("renders", {})
    return [{**row, "duration_seconds": segments[row["segment_id"]]["duration_seconds"], "start_seconds": segments[row["segment_id"]]["start_seconds"], "end_seconds": segments[row["segment_id"]]["end_seconds"], "render_status": renders.get(row["segment_id"], "ready")} for row in rows]


def render_tagged_clip(session_id: str, segment_id: str, label: str) -> None:
    """Regera um único MP4 em segundo plano após uma classificação."""
    state = session_or_404(session_id)
    output = Path(state["output"])
    try:
        segment = next(item for item in read_csv(output / "segments.csv") if item["segment_id"] == segment_id)
        filename = output_name(segment_id, label)
        render_clip(Path(state["video"]), output / "clips" / filename, float(segment["start_seconds"]), float(segment["end_seconds"]), label, evidence_text(segment))
        with annotation_lock:
            update_manifest_filename(output, segment_id, filename)
            state["renders"][segment_id] = "ready"
        print(f"[{time.strftime('%H:%M:%S')}] Clipe {segment_id} classificado como {label}.", flush=True)
    except Exception as error:
        state["renders"][segment_id] = "failed"
        print(f"[{time.strftime('%H:%M:%S')}] ERRO ao marcar {segment_id}: {error}", flush=True)


@app.patch("/sessions/{session_id}/clips/{segment_id}")
def tag(session_id: str, segment_id: str, request: TagRequest) -> dict[str, object]:
    if request.tag not in {"GOAL", "NO_GOAL", "NOT_EVALUABLE"}:
        raise HTTPException(400, "Tag inválida.")
    state = session_or_404(session_id)
    output = Path(state["output"])
    with annotation_lock:
        rows, fields = load_annotation_rows(output / "clips" / "clips_manifest.csv", output / "clips" / "clip_annotations.csv")
        for row in rows:
            if row["segment_id"] == segment_id:
                row.update({"human_label": request.tag, "annotation_status": "completed", "notes": request.notes})
                write_annotation_rows(output / "clips" / "clip_annotations.csv", fields, rows)
                update_manifest_label(output, segment_id, request.tag)
                state["renders"][segment_id] = "rendering"
                threading.Thread(target=render_tagged_clip, args=(session_id, segment_id, request.tag), daemon=True).start()
                return {"ok": True, "clip_filename": row.get("clip_filename", ""), "render_status": "rendering"}
    raise HTTPException(404, "Clipe não encontrado.")


@app.get("/sessions/{session_id}/clips/{segment_id}/download")
def download(session_id: str, segment_id: str) -> FileResponse:
    state = session_or_404(session_id)
    rows, _ = session_rows(state)
    row = next((item for item in rows if item["segment_id"] == segment_id), None)
    if not row:
        raise HTTPException(404, "Clipe não encontrado.")
    path = ensure_export_overlay(state, row)
    return FileResponse(path, filename=output_name(segment_id, row.get("human_label")))


@app.get("/sessions/{session_id}/clips/{segment_id}/media")
def media(session_id: str, segment_id: str) -> FileResponse:
    """Serve o MP4 H.264 sem anexo para o elemento video do Electron."""
    state = session_or_404(session_id)
    rows, _ = session_rows(state)
    row = next((item for item in rows if item["segment_id"] == segment_id), None)
    if not row:
        raise HTTPException(404, "Clipe não encontrado.")
    path = Path(state["output"]) / "clips" / row["clip_filename"]
    if not path.is_file():
        raise HTTPException(404, "Arquivo do clipe não encontrado.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/sessions/{session_id}/clips/compilation")
def compilation(session_id: str, segment_ids: list[str] = Query()) -> FileResponse:
    if not segment_ids:
        raise HTTPException(400, "Selecione ao menos um clipe.")
    state = session_or_404(session_id)
    rows, _ = session_rows(state)
    selected = [row for row in rows if row["segment_id"] in segment_ids]
    if not selected:
        raise HTTPException(404, "Clipes não encontrados.")
    selected.sort(key=lambda row: clip_number(row["segment_id"]))
    numbers = [clip_number(row["segment_id"]) for row in selected]
    filename = f"clipes_{numbers[0]}-{numbers[-1]}_compilation.mp4"
    destination = Path(state["workspace"]) / filename
    compile_clips([ensure_export_overlay(state, row) for row in selected], destination)
    return FileResponse(destination, filename=filename)


@app.get("/sessions/{session_id}/source")
def source(session_id: str) -> FileResponse:
    return FileResponse(Path(session_or_404(session_id)["video"]), media_type="video/mp4")


@app.delete("/sessions/{session_id}")
def close(session_id: str) -> dict[str, bool]:
    state = session_or_404(session_id)
    shutil.rmtree(Path(state["workspace"]), ignore_errors=True)
    del sessions[session_id]
    return {"ok": True}
