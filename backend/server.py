"""API local para o aplicativo Electron de análise de lances."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
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
from app.review_storage import load_annotation_rows, read_csv, read_json, write_annotation_rows, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
SESSION_ROOT = ROOT / "outputs" / "review_app" / "sessions"
app = FastAPI(title="Goal Event Review API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
sessions: dict[str, dict[str, object]] = {}
annotation_lock = threading.Lock()
session_persist_lock = threading.Lock()
telemetry_lock = threading.Lock()
compilation_jobs: dict[str, dict[str, object]] = {}
compilation_lock = threading.Lock()
gpu_utilization_supported: bool | None = None


class StartRequest(BaseModel):
    video_path: str
    mode: str = "alta_cobertura"


class TagRequest(BaseModel):
    tag: str | None = None
    notes: str | None = None
    review_status: str | None = None


class SessionStateRequest(BaseModel):
    current_segment_id: str | None = None
    current_time: float | None = None
    review_filter: str | None = None
    auto_advance: bool | None = None
    video_mode: str | None = None


class TelemetryRequest(BaseModel):
    event: str
    candidate_id: str | None = None
    value: float | str | None = None


class CompilationRequest(BaseModel):
    segment_ids: list[str]


def persist_session(state: dict[str, object]) -> None:
    """Persiste estado operacional e retomada sem expor caminhos na API."""
    path = Path(state["workspace"]) / "session.json"
    with session_persist_lock:
        serializable = {key: value for key, value in state.items() if key not in {"_thread", "_cancel_event"}}
        write_json_atomic(path, serializable)


def restore_sessions() -> None:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    for metadata_path in SESSION_ROOT.glob("*/session.json"):
        try:
            state = read_json(metadata_path)
            session_id = str(state.get("session_id", ""))
            workspace = metadata_path.parent.resolve()
            if not session_id or workspace.parent != SESSION_ROOT.resolve() or workspace.name != session_id:
                continue
            state["workspace"] = str(workspace)
            state["output"] = str(workspace / "analysis")
            state.setdefault("video_mode", "clip")
            state.setdefault("undo_history", [])
            state["_cancel_event"] = threading.Event()
            renders_changed = False
            for segment_id, render_status in dict(state.get("renders", {})).items():
                if render_status == "rendering":
                    manifest = workspace / "analysis" / "clips" / "clips_manifest.csv"
                    matching = next((row for row in read_csv(manifest) if row.get("segment_id") == segment_id), None) if manifest.is_file() else None
                    rendered = matching and (manifest.parent / matching.get("clip_filename", "")).is_file()
                    state["renders"][segment_id] = "ready" if rendered else "failed"
                    renders_changed = True
            if state.get("status") == "cancelling":
                shutil.rmtree(workspace / "analysis", ignore_errors=True)
                state.update({"status": "cancelled", "stage": "Análise cancelada", "error": None})
                persist_session(state)
            elif state.get("status") in {"running", "starting"}:
                state.update({"status": "interrupted", "stage": "Análise interrompida", "error": "O aplicativo foi fechado durante o processamento. Inicie uma nova análise para processar este vídeo novamente."})
                persist_session(state)
            elif renders_changed:
                persist_session(state)
            sessions[session_id] = state
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Não foi possível restaurar a análise {metadata_path.parent.name}: {error}", flush=True)


@app.on_event("startup")
def load_persisted_sessions() -> None:
    restore_sessions()


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


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_paths() -> dict[str, Path]:
    models = ROOT / "experiments/ball"
    return {
        "goal_detector": ROOT / "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt",
        "ball_detector": ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt",
        "temporal_ranker": models / "GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib",
        "visual_verifier": models / "GOAL-EVENT-PIPELINE-FINAL-V1/temporal_sequence_verifier.joblib",
        "fusion_model": models / "GOAL-EVENT-FUSION-FINAL-V1/fusion_reranker.joblib",
    }


def device_metrics() -> dict[str, object]:
    global gpu_utilization_supported
    process = psutil.Process(os.getpid())
    metrics: dict[str, object] = {
        "processor": "GPU" if torch.cuda.is_available() else "CPU",
        "process_cpu_percent": round(process.cpu_percent(interval=None), 1),
        "process_ram_mb": round(process.memory_info().rss / 1024**2, 1),
        "process_ram_percent": round(process.memory_percent(), 2),
        "cpu_name": platform.processor() or platform.uname().processor or "CPU do sistema",
    }
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            used_bytes = total_bytes - free_bytes
            metrics.update({
                "device_name": torch.cuda.get_device_name(0),
                "gpu_memory_used_mb": round(used_bytes / 1024**2, 1),
                "gpu_memory_total_mb": round(total_bytes / 1024**2, 1),
                "gpu_memory_percent": round(used_bytes / total_bytes * 100, 1),
                "gpu_process_memory_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
            })
            if gpu_utilization_supported is not False:
                try:
                    metrics["gpu_utilization_percent"] = torch.cuda.utilization(0)
                    gpu_utilization_supported = True
                except Exception:
                    gpu_utilization_supported = False
            if gpu_utilization_supported is False:
                metrics["gpu_utilization_percent"] = None
        except Exception as error:
            metrics["gpu_metrics_error"] = str(error)
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


def session_rows(state: dict[str, object]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    output = Path(state["output"])
    manifest = output / "clips" / "clips_manifest.csv"
    segments_path = output / "segments.csv"
    if not manifest.is_file() or not segments_path.is_file():
        raise HTTPException(409, "Os clipes ainda estão sendo preparados.")
    rows, _ = load_annotation_rows(manifest, output / "clips" / "clip_annotations.csv")
    segments = {row["segment_id"]: row for row in read_csv(segments_path)}
    return rows, segments


def selected_candidate_ranks(output: Path) -> dict[str, int]:
    path = output / "selected_candidates.csv"
    if not path.is_file():
        return {}
    return {row.get("event_id", ""): index for index, row in enumerate(read_csv(path), start=1) if row.get("event_id")}


def ensure_clean_clip(state: dict[str, object], row: dict[str, str]) -> Path:
    """Garante que o MP4 não tenha o overlay legado de classificação."""
    output = Path(state["output"])
    manifest = output / "clips" / "clips_manifest.csv"
    with annotation_lock:
        manifest_rows = read_csv(manifest)
        manifest_row = next((item for item in manifest_rows if item["segment_id"] == row["segment_id"]), None)
        if manifest_row is None:
            raise HTTPException(404, "Clipe não encontrado.")
        current = output / "clips" / manifest_row["clip_filename"]
        if manifest_row.get("video_overlay") == "none" and current.is_file():
            return current
        source = Path(state["video"])
        if not source.is_file():
            raise HTTPException(409, "Este clipe ainda tem o overlay antigo e o vídeo original não está disponível para removê-lo.")
        segment = next((item for item in read_csv(output / "segments.csv") if item["segment_id"] == row["segment_id"]), None)
        if segment is None:
            raise HTTPException(404, "Segmento do clipe não encontrado.")
        render_clip(source, current, float(segment["start_seconds"]), float(segment["end_seconds"]))
        for item in manifest_rows:
            item["video_overlay"] = "none"
        write_rows(manifest, manifest_rows)
        return current


def compile_clips(paths: list[Path], destination: Path, progress=None, job=None) -> None:
    if not FFMPEG.is_file():
        raise RuntimeError("FFmpeg local não encontrado.")
    list_path = destination.with_suffix(".txt")
    entries = ["file '{}'\n".format(path.as_posix().replace("'", "'\\\\''")) for path in paths]
    list_path.write_text("".join(entries), encoding="utf-8")
    try:
        command = [
            str(FFMPEG), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(destination),
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if job is not None:
            job["process"] = process
        duration = sum(probe_duration(path) for path in paths)
        while process.poll() is None:
            if job is not None and job.get("cancel_requested"):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise InterruptedError("Compilação cancelada.")
            line = process.stdout.readline() if process.stdout else ""
            if line.startswith("out_time_ms=") and progress and duration > 0:
                try:
                    progress(min(99.0, int(line.split("=", 1)[1]) / 1_000_000 / duration * 100))
                except (ValueError, IndexError):
                    pass
            elif not line:
                time.sleep(0.05)
        stderr = process.stderr.read() if process.stderr else ""
        if process.returncode:
            raise RuntimeError(f"Não foi possível criar a compilação: {stderr[-500:]}")
    finally:
        list_path.unlink(missing_ok=True)


def probe_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return frames / fps if fps > 0 else 0.0
    finally:
        capture.release()


def run_compilation(session_id: str, job_id: str, segment_ids: list[str]) -> None:
    job = compilation_jobs[job_id]
    try:
        state = session_or_404(session_id)
        rows, _ = session_rows(state)
        selected_rows = [row for row in rows if row["segment_id"] in segment_ids]
        if not selected_rows:
            raise RuntimeError("Os clipes selecionados não foram encontrados.")
        selected_rows.sort(key=lambda row: int(clip_number(row["segment_id"])))
        job.update({"status": "preparing", "stage": "Preparando clipes", "progress": 0.0})
        paths = []
        for index, row in enumerate(selected_rows, start=1):
            if job.get("cancel_requested"):
                raise InterruptedError("Compilação cancelada.")
            paths.append(ensure_clean_clip(state, row))
            job.update({"progress": min(20.0, index / len(selected_rows) * 20.0)})
        numbers = [clip_number(row["segment_id"]) for row in selected_rows]
        filename = f"clipes_{numbers[0]}-{numbers[-1]}_compilation.mp4"
        destination = Path(state["workspace"]) / filename
        job.update({"status": "compiling", "stage": "Montando vídeo", "progress": 20.0, "filename": filename, "path": str(destination)})
        compile_clips(paths, destination, lambda value: job.update({"progress": 20.0 + value * 0.8}), job)
        if job.get("cancel_requested"):
            raise InterruptedError("Compilação cancelada.")
        job.update({"status": "ready", "stage": "Pronto para baixar", "progress": 100.0})
    except InterruptedError:
        if job.get("path"):
            Path(str(job["path"])).unlink(missing_ok=True)
        job.update({"status": "cancelled", "stage": "Cancelada"})
    except Exception as error:
        if job.get("path"):
            Path(str(job["path"])).unlink(missing_ok=True)
        job.update({"status": "failed", "stage": "Falha", "error": str(error)})


def run_session(session_id: str, request: StartRequest) -> None:
    state = session_or_404(session_id)
    video = Path(request.video_path)
    output = Path(state["output"])
    cancel_event = state["_cancel_event"]
    started = time.perf_counter()
    try:
        match_id = re.sub(r"[^a-zA-Z0-9_]+", "_", video.stem).strip("_") or "video"

        def check_cancelled() -> None:
            if cancel_event.is_set():
                raise InterruptedError("Análise cancelada pelo usuário.")

        def progress(label: str, value: float) -> None:
            check_cancelled()
            elapsed = time.perf_counter() - started
            percentage = max(0.0, min(float(value), 100.0))
            estimate = elapsed / percentage * 100.0 if percentage >= 1.0 else None
            state.update({"stage": label, "progress": percentage, "elapsed": elapsed, "estimated_total": estimate, "metrics": device_metrics()})
            persist_session(state)
            print(f"[{time.strftime('%H:%M:%S')}] {label} ({percentage:.2f}%)", flush=True)

        models = model_paths()
        check_cancelled()
        analyze_main([
            "--video-path", str(video), "--match-id", match_id, "--mode", request.mode,
            "--output", str(output), "--videos-dir", str(video.parent), "--device", "cuda:0",
            "--allow-arbitrary-video", "--split-manifest", str(ROOT / "experiments/datasets/dataset_split_v1.csv"),
            "--goal-model", str(models["goal_detector"]),
            "--ball-model", str(models["ball_detector"]),
            "--temporal-ranker", str(models["temporal_ranker"]),
            "--visual-verifier", str(models["visual_verifier"]),
            "--fusion-model", str(models["fusion_model"]),
        ], progress, check_cancelled)
        check_cancelled()
        clips_main(["--segments", str(output / "segments.csv"), "--video", str(video), "--output", str(output / "clips")], progress, check_cancelled)
        check_cancelled()
        print(f"[{time.strftime('%H:%M:%S')}] Análise concluída.", flush=True)
        state.update({"stage": "Concluído", "progress": 100.0, "status": "completed", "elapsed": time.perf_counter() - started, "metrics": device_metrics()})
        persist_session(state)
    except InterruptedError:
        shutil.rmtree(output, ignore_errors=True)
        state.update({"status": "cancelled", "stage": "Análise cancelada", "error": None, "estimated_total": None, "elapsed": time.perf_counter() - started, "metrics": device_metrics()})
        persist_session(state)
    except Exception as error:
        print(f"[{time.strftime('%H:%M:%S')}] ERRO: {error}", flush=True)
        state.update({"status": "failed", "error": str(error), "elapsed": time.perf_counter() - started, "metrics": device_metrics()})
        persist_session(state)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.4.0"}


@app.post("/sessions")
def start(request: StartRequest) -> dict[str, str]:
    if not Path(request.video_path).is_file():
        raise HTTPException(400, "Vídeo não encontrado.")
    from src.events.operational_modes_v4 import MODES as PIPELINE_MODES
    if request.mode not in PIPELINE_MODES:
        raise HTTPException(400, "Modo operacional inválido.")
    video_capture = cv2.VideoCapture(request.video_path)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS) if video_capture.isOpened() else 0.0
    video_frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) if video_capture.isOpened() else 0.0
    video_capture.release()
    video_duration = video_frame_count / video_fps if video_fps > 0 and video_frame_count > 0 else None
    from src.events.operational_modes_v4 import MODES as OPERATIONAL_MODE_CONFIG
    config_hash = hashlib.sha256((ROOT / "src/events/operational_modes_v4.py").read_bytes()).hexdigest()
    model_hashes = {name: sha256_file(path) for name, path in model_paths().items()}
    session_id = uuid.uuid4().hex
    workspace = SESSION_ROOT / session_id
    workspace.mkdir(parents=True, exist_ok=False)
    sessions[session_id] = {
        "workspace": str(workspace), "output": str(workspace / "analysis"), "video": request.video_path,
        "session_id": session_id, "video_name": Path(request.video_path).name, "mode": request.mode,
        "created_at": datetime.now(timezone.utc).isoformat(), "app_version": "0.4.0", "pipeline_version": "V4",
        "video_fps": video_fps if video_fps > 0 else None,
        "video_duration_seconds": video_duration,
        "mode_config_hash": config_hash, "mode_description": OPERATIONAL_MODE_CONFIG[request.mode], "model_hashes": model_hashes,
        "status": "running", "stage": "Iniciando análise", "progress": 0.0, "elapsed": 0.0,
        "estimated_total": None, "metrics": device_metrics(), "renders": {},
        "undo_history": [], "_cancel_event": threading.Event(),
        "current_segment_id": None, "current_time": 0.0, "review_filter": "all", "auto_advance": False,
    }
    persist_session(sessions[session_id])
    threading.Thread(target=run_session, args=(session_id, request), daemon=True).start()
    return {"session_id": session_id}


@app.get("/sessions/{session_id}")
def status(session_id: str) -> dict[str, object]:
    state = session_or_404(session_id)
    state["metrics"] = device_metrics()
    visible = {key: value for key, value in state.items() if key not in {"output", "workspace", "video", "_thread", "_cancel_event", "undo_history"}}
    visible["can_undo"] = bool(state.get("undo_history", []))
    visible["video_available"] = Path(str(state.get("video", ""))).is_file()
    return visible


@app.post("/sessions/{session_id}/cancel")
def cancel_analysis(session_id: str) -> dict[str, object]:
    state = session_or_404(session_id)
    if state.get("status") == "cancelling":
        return {"ok": True, "status": "cancelling"}
    if state.get("status") not in {"running", "starting"}:
        raise HTTPException(409, "Esta análise não está em processamento.")
    cancel_event = state.get("_cancel_event")
    if cancel_event is None:
        raise HTTPException(409, "Não foi possível acessar o controle de cancelamento desta análise.")
    cancel_event.set()
    state.update({"status": "cancelling", "stage": "Cancelando análise"})
    persist_session(state)
    return {"ok": True, "status": "cancelling"}


@app.get("/sessions")
def list_sessions() -> list[dict[str, object]]:
    summaries = []
    for state in sessions.values():
        output = Path(state["output"])
        reviewed = 0
        annotations = output / "clips" / "clip_annotations.csv"
        if annotations.is_file():
            try:
                reviewed = sum(row.get("annotation_status") == "completed" for row in read_csv(annotations))
            except (OSError, KeyError, csv.Error):
                reviewed = 0
        metrics = state.get("metrics") or {}
        summaries.append({key: state.get(key) for key in ("session_id", "video_name", "mode", "created_at", "status", "stage", "progress", "error", "elapsed", "video_duration_seconds")})
        summaries[-1]["metrics"] = metrics
        summaries[-1]["reviewed_count"] = reviewed
    return sorted(summaries, key=lambda item: str(item.get("created_at", "")), reverse=True)


@app.patch("/sessions/{session_id}/state")
def update_session_state(session_id: str, request: SessionStateRequest) -> dict[str, bool]:
    state = session_or_404(session_id)
    if request.review_filter is not None and request.review_filter not in {"all", "pending", "deferred", "completed"}:
        raise HTTPException(400, "Filtro de revisão inválido.")
    if request.current_time is not None and request.current_time < 0:
        raise HTTPException(400, "A posição do vídeo não pode ser negativa.")
    if request.video_mode is not None and request.video_mode not in {"clip", "source"}:
        raise HTTPException(400, "Modo de reprodução inválido.")
    if request.current_segment_id is not None:
        output = Path(state["output"])
        manifest = output / "clips" / "clips_manifest.csv"
        if not manifest.is_file() or request.current_segment_id not in {row["segment_id"] for row in read_csv(manifest)}:
            raise HTTPException(404, "Clipe não encontrado.")
    for key, value in request.model_dump(exclude_unset=True).items():
        state[key] = value
    persist_session(state)
    return {"ok": True}


@app.post("/sessions/{session_id}/telemetry")
def record_telemetry(session_id: str, request: TelemetryRequest) -> dict[str, bool]:
    state = session_or_404(session_id)
    allowed = {"review_started", "review_finished", "candidate_opened", "decision_saved", "review_deferred", "play", "pause", "seek", "rate_change", "storyboard_used", "focus_mode", "export"}
    if request.event not in allowed:
        raise HTTPException(400, "Evento de telemetria inválido.")
    path = Path(state["workspace"]) / "review_telemetry.jsonl"
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": request.event, "candidate_id": request.candidate_id, "value": request.value}
    with telemetry_lock, path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/sessions/{session_id}/clips")
def clips(session_id: str) -> list[dict[str, object]]:
    state = session_or_404(session_id)
    output = Path(state["output"])
    if not (output / "clips" / "clips_manifest.csv").is_file() or not (output / "segments.csv").is_file():
        if state.get("status") in {"running", "interrupted", "failed"}:
            return []
        raise HTTPException(409, "A análise foi concluída sem manifesto de clipes.")
    rows, segments = session_rows(state)
    renders = state.get("renders", {})
    ranks = selected_candidate_ranks(Path(state["output"]))
    result = []
    for position, row in enumerate(rows, start=1):
        segment = segments.get(row["segment_id"], {})
        candidate_id = segment.get("representative_event_id") or row["segment_id"]
        result.append({
            **row,
            "candidate_id": candidate_id,
            "rank": ranks.get(candidate_id, position),
            "review_position": position,
            "timestamp_seconds": segment.get("representative_timestamp_seconds", ""),
            "timestamp": segment.get("representative_time_hhmmss", ""),
            "rank_score": segment.get("rank_score", ""),
            "score": segment.get("fusion_score_goal", ""),
            "method": segment.get("method", segment.get("source", "")),
            "duration_seconds": segment.get("duration_seconds", ""),
            "member_count": segment.get("member_count", "1"),
            "start_seconds": segment.get("start_seconds", ""),
            "end_seconds": segment.get("end_seconds", ""),
            "render_status": renders.get(row["segment_id"], "ready"),
        })
    return sorted(result, key=lambda item: (int(item.get("rank") or 0), float(item.get("timestamp_seconds") or 0)))


@app.get("/sessions/{session_id}/clips/{segment_id}/storyboard")
def storyboard(session_id: str, segment_id: str) -> list[dict[str, object]]:
    state = session_or_404(session_id)
    source = Path(state["video"])
    if not source.is_file():
        raise HTTPException(404, "Vídeo original não está disponível.")
    segments = {row["segment_id"]: row for row in read_csv(Path(state["output"]) / "segments.csv")}
    segment = segments.get(segment_id)
    if not segment:
        raise HTTPException(404, "Candidato não encontrado.")
    start = float(segment["start_seconds"])
    end = float(segment["end_seconds"])
    candidate_time = float(segment.get("representative_timestamp_seconds") or (start + end) / 2)
    stat = source.stat()
    cache = Path(state["workspace"]) / "storyboards"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise HTTPException(415, "Não foi possível ler o vídeo original para gerar o storyboard.")
        for index, offset in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0)):
            source_time = max(0.0, min(candidate_time + offset, end))
            clip_time = max(0.0, min(source_time - start, end - start))
            frame_path = cache / f"{segment_id}_{stat.st_size}_{stat.st_mtime_ns}_{index}.jpg"
            if not frame_path.is_file():
                capture.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000)
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                if width > 480:
                    frame = cv2.resize(frame, (480, max(1, round(height * 480 / width))), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 82]):
                    continue
            frames.append({
                "offset_seconds": offset,
                "timestamp_seconds": source_time,
                "clip_time_seconds": clip_time,
                "image_url": f"/sessions/{session_id}/clips/{segment_id}/storyboard/frame/{index}",
            })
    finally:
        capture.release()
    return frames


@app.get("/sessions/{session_id}/clips/{segment_id}/storyboard/frame/{index}")
def storyboard_frame(session_id: str, segment_id: str, index: int) -> FileResponse:
    if index not in range(5):
        raise HTTPException(404, "Quadro do storyboard não encontrado.")
    state = session_or_404(session_id)
    source = Path(state["video"])
    if not source.is_file():
        raise HTTPException(404, "Vídeo original não está disponível.")
    stat = source.stat()
    path = Path(state["workspace"]) / "storyboards" / f"{segment_id}_{stat.st_size}_{stat.st_mtime_ns}_{index}.jpg"
    if not path.is_file():
        raise HTTPException(404, "Gere o storyboard antes de abrir este quadro.")
    return FileResponse(path, media_type="image/jpeg")


@app.patch("/sessions/{session_id}/clips/{segment_id}")
def tag(session_id: str, segment_id: str, request: TagRequest) -> dict[str, object]:
    if request.tag is not None and request.tag not in {"GOAL", "NO_GOAL", "NOT_EVALUABLE"}:
        raise HTTPException(400, "Tag inválida.")
    if request.review_status is not None and request.review_status not in {"pending", "deferred", "completed"}:
        raise HTTPException(400, "Estado de revisão inválido.")
    if request.tag is None and request.notes is None and request.review_status is None:
        raise HTTPException(400, "Informe uma decisão, nota ou estado de revisão.")
    state = session_or_404(session_id)
    output = Path(state["output"])
    with annotation_lock:
        rows, fields = load_annotation_rows(output / "clips" / "clips_manifest.csv", output / "clips" / "clip_annotations.csv")
        for row in rows:
            if row["segment_id"] == segment_id:
                undo_action = None
                if request.tag is not None:
                    previous_label = row.get("human_label", "")
                    previous_status = row.get("annotation_status", "pending")
                    row["human_label"] = request.tag
                    row["annotation_status"] = "completed"
                    if (previous_label, previous_status) != (request.tag, "completed"):
                        undo_action = {
                            "segment_id": segment_id,
                            "before": {"human_label": previous_label, "annotation_status": previous_status},
                            "after": {"human_label": request.tag, "annotation_status": "completed"},
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                if request.review_status is not None:
                    if request.review_status == "completed" and not (request.tag or row.get("human_label")):
                        raise HTTPException(400, "Uma revisão sem rótulo humano não pode ser marcada como concluída.")
                    if request.review_status in {"pending", "deferred"} and row.get("human_label"):
                        raise HTTPException(400, "Remova ou atualize o rótulo humano antes de alterar o fluxo da revisão.")
                    row["annotation_status"] = request.review_status
                if request.notes is not None:
                    row["notes"] = request.notes
                write_annotation_rows(output / "clips" / "clip_annotations.csv", fields, rows)
                if undo_action is not None:
                    history = state.setdefault("undo_history", [])
                    history.append(undo_action)
                    persist_session(state)
                return {"ok": True, "clip_filename": row.get("clip_filename", ""), "render_status": "ready", "can_undo": bool(state.get("undo_history", []))}
    raise HTTPException(404, "Clipe não encontrado.")


@app.post("/sessions/{session_id}/undo")
def undo_last_decision(session_id: str) -> dict[str, object]:
    state = session_or_404(session_id)
    with annotation_lock:
        history = state.setdefault("undo_history", [])
        if not history:
            raise HTTPException(409, "Não há decisão recente para desfazer.")
        action = history[-1]
        segment_id = str(action.get("segment_id", ""))
        output = Path(state["output"])
        rows, fields = load_annotation_rows(output / "clips" / "clips_manifest.csv", output / "clips" / "clip_annotations.csv")
        row = next((item for item in rows if item.get("segment_id") == segment_id), None)
        if row is None:
            raise HTTPException(404, "O clipe da última decisão não foi encontrado.")
        before = action.get("before", {})
        after = action.get("after", {})
        current = {"human_label": row.get("human_label", ""), "annotation_status": row.get("annotation_status", "pending")}
        if current != after:
            raise HTTPException(409, "A decisão mudou depois da última ação; atualize a sessão antes de desfazer.")
        row.update(before)
        write_annotation_rows(output / "clips" / "clip_annotations.csv", fields, rows)
        history.pop()
        persist_session(state)
        return {"ok": True, "segment_id": segment_id, "restored": before, "can_undo": bool(history)}


@app.get("/sessions/{session_id}/export")
def export_review(session_id: str, format: str = Query(pattern="^(csv|json)$")) -> FileResponse:
    state = session_or_404(session_id)
    rows, segments = session_rows(state)
    ranks = selected_candidate_ranks(Path(state["output"]))
    export_rows = []
    for position, row in enumerate(rows, start=1):
        segment = segments.get(row["segment_id"], {})
        candidate_id = segment.get("representative_event_id") or row["segment_id"]
        export_rows.append({
            "candidate_id": candidate_id,
            "segment_id": row["segment_id"],
            "timestamp_seconds": segment.get("representative_timestamp_seconds", ""),
            "timestamp": segment.get("representative_time_hhmmss", ""),
            "rank": ranks.get(candidate_id, position),
            "review_position": position,
            "rank_score": segment.get("rank_score", ""),
            "score": segment.get("fusion_score_goal", ""),
            "human_label": row.get("human_label", ""),
            "review_status": row.get("annotation_status", "pending"),
            "mode": state.get("mode", ""),
            "segment_start": segment.get("start_seconds", ""),
            "segment_end": segment.get("end_seconds", ""),
            "notes": row.get("notes", ""),
        })
    destination = Path(state["workspace"]) / f"revisao.{format}"
    if format == "json":
        write_json_atomic(destination, {
            "session_id": state.get("session_id"), "video_name": state.get("video_name"),
            "mode": state.get("mode"), "created_at": state.get("created_at"),
            "app_version": state.get("app_version"), "pipeline_version": state.get("pipeline_version"),
            "mode_config_hash": state.get("mode_config_hash"), "model_hashes": state.get("model_hashes", {}),
            "candidates": export_rows,
        })
        media_type = "application/json"
    else:
        field_names = {
            "candidate_id": "ID do candidato", "segment_id": "ID do segmento", "timestamp_seconds": "Tempo na partida (segundos)",
            "timestamp": "Tempo na partida", "rank": "Posição no ranking", "review_position": "Ordem de revisão",
            "rank_score": "Pontuação do ranking", "score": "Pontuação do modelo", "human_label": "Decisão humana",
            "review_status": "Estado da revisão", "mode": "Modo de análise", "segment_start": "Início do clipe (segundos)",
            "segment_end": "Fim do clipe (segundos)", "notes": "Anotações",
        }
        decision_names = {"GOAL": "Possível gol", "NO_GOAL": "Sem gol", "NOT_EVALUABLE": "Não avaliável"}
        review_names = {"completed": "Concluída", "deferred": "Revisar depois", "pending": "Pendente"}
        mode_names = {"alta_cobertura": "Alta cobertura", "cobertura_reduzida": "Cobertura reduzida", "triagem_rapida": "Triagem rápida"}
        fields = list(export_rows[0]) if export_rows else list(field_names)
        with destination.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writerow({key: field_names.get(key, key) for key in fields})
            for row in export_rows:
                localized = dict(row)
                localized["human_label"] = decision_names.get(str(row.get("human_label")), row.get("human_label", ""))
                localized["review_status"] = review_names.get(str(row.get("review_status")), row.get("review_status", ""))
                localized["mode"] = mode_names.get(str(row.get("mode")), row.get("mode", ""))
                writer.writerow(localized)
        media_type = "text/csv"
    return FileResponse(destination, media_type=media_type, filename=destination.name)


def review_metrics(state: dict[str, object], rows: list[dict[str, str]], segments: dict[str, dict[str, str]]) -> dict[str, float | int]:
    events_path = Path(state["workspace"]) / "review_telemetry.jsonl"
    events = []
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    def paired_seconds(start_event: str, end_event: str) -> float:
        started = None
        total = 0.0
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"]).timestamp()
            except (KeyError, TypeError, ValueError):
                continue
            if event.get("event") == start_event:
                started = timestamp
            elif event.get("event") == end_event and started is not None:
                total += max(0.0, timestamp - started)
                started = None
        return total
    intervals = sorted((float(item["start_seconds"]), float(item["end_seconds"])) for item in segments.values() if item.get("start_seconds") and item.get("end_seconds"))
    unique_seconds = 0.0
    if intervals:
        left, right = intervals[0]
        for start, end in intervals[1:]:
            if start <= right:
                right = max(right, end)
            else:
                unique_seconds += max(0.0, right - left)
                left, right = start, end
        unique_seconds += max(0.0, right - left)
    return {
        "review_elapsed_seconds": round(paired_seconds("review_started", "review_finished"), 1),
        "active_playback_seconds": round(paired_seconds("play", "pause"), 1),
        "seek_count": sum(event.get("event") == "seek" for event in events),
        "candidate_revisits": max(0, sum(event.get("event") == "candidate_opened" for event in events) - len({event.get("candidate_id") for event in events if event.get("event") == "candidate_opened"})),
        "candidate_count": len(rows),
        "completed_count": sum(row.get("annotation_status") == "completed" for row in rows),
        "deferred_count": sum(row.get("annotation_status") == "deferred" for row in rows),
        "pending_count": sum(row.get("annotation_status") == "pending" for row in rows),
        "video_minutes_delivered_unique": round(unique_seconds / 60, 2),
    }


@app.get("/sessions/{session_id}/report.html")
def review_report(session_id: str) -> FileResponse:
    state = session_or_404(session_id)
    rows, segments = session_rows(state)
    metrics = review_metrics(state, rows, segments)
    safe_name = html.escape(str(state.get("video_name", "Vídeo")))
    mode_names = {"alta_cobertura": "Alta cobertura", "cobertura_reduzida": "Cobertura reduzida", "triagem_rapida": "Triagem rápida"}
    safe_mode = html.escape(mode_names.get(str(state.get("mode", "")), str(state.get("mode", ""))))
    metric_names = {
        "review_elapsed_seconds": "Tempo entre abrir e encerrar a revisão (segundos)",
        "active_playback_seconds": "Tempo com o vídeo em reprodução (segundos)",
        "seek_count": "Buscas na linha do tempo",
        "candidate_revisits": "Candidatos revisitados",
        "candidate_count": "Clipes disponíveis para revisão",
        "completed_count": "Clipes com decisão registrada",
        "deferred_count": "Clipes marcados para revisar depois",
        "pending_count": "Clipes ainda pendentes",
        "video_minutes_delivered_unique": "Minutos únicos de vídeo apresentados",
    }
    metric_rows = "".join(f"<tr><th>{html.escape(metric_names.get(key, key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in metrics.items())
    hardware = state.get("metrics") or {}
    hardware_name = hardware.get("device_name") if hardware.get("processor") == "GPU" else hardware.get("cpu_name")
    elapsed_seconds = max(0, int(float(state.get("elapsed") or 0)))
    elapsed = f"{elapsed_seconds // 3600:02d}:{(elapsed_seconds % 3600) // 60:02d}:{elapsed_seconds % 60:02d}"
    destination = Path(state["workspace"]) / "relatorio_revisao.html"
    destination.write_text(
        "<!doctype html><html lang='pt-BR'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>Relatório de revisão — {safe_name}</title><style>body{{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px;color:#20332d}}"
        "table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #ddd;padding:10px}small{color:#66766d}</style>"
        f"<h1>Relatório de revisão assistida</h1><p><b>Vídeo:</b> {safe_name}<br><b>Modo:</b> {safe_mode}<br>"
        f"<b>Iniciada:</b> {html.escape(str(state.get('created_at', '')))}<br><b>Duração do processamento:</b> {html.escape(elapsed)}<br><b>Equipamento usado:</b> {html.escape(str(hardware_name or 'Não informado'))}<br><b>Versão da aplicação/pipeline:</b> {html.escape(str(state.get('app_version', '')))} / {html.escape(str(state.get('pipeline_version', '')))}</p><h2>Resumo da revisão</h2><table>{metric_rows}</table>"
        "<p><small>As decisões são anotações humanas. Os minutos únicos medem quanto tempo diferente da partida foi incluído nos clipes; não são o tempo gasto pela pessoa e não confirmam gols. Os arquivos CSV e JSON guardam os registros detalhados para análise.</small></p></html>",
        encoding="utf-8",
    )
    return FileResponse(destination, media_type="text/html", filename=destination.name)


@app.get("/sessions/{session_id}/clips/{segment_id}/download")
def download(session_id: str, segment_id: str) -> FileResponse:
    state = session_or_404(session_id)
    rows, _ = session_rows(state)
    row = next((item for item in rows if item["segment_id"] == segment_id), None)
    if not row:
        raise HTTPException(404, "Clipe não encontrado.")
    path = ensure_clean_clip(state, row)
    return FileResponse(path, filename=output_name(segment_id, row.get("human_label")))


@app.get("/sessions/{session_id}/clips/{segment_id}/media")
def media(session_id: str, segment_id: str) -> FileResponse:
    """Serve o MP4 H.264 sem anexo para o elemento video do Electron."""
    state = session_or_404(session_id)
    rows, _ = session_rows(state)
    row = next((item for item in rows if item["segment_id"] == segment_id), None)
    if not row:
        raise HTTPException(404, "Clipe não encontrado.")
    path = ensure_clean_clip(state, row)
    return FileResponse(path, media_type="video/mp4")


@app.post("/sessions/{session_id}/compilations")
def start_compilation(session_id: str, request: CompilationRequest) -> dict[str, str]:
    session_or_404(session_id)
    if not request.segment_ids:
        raise HTTPException(400, "Selecione ao menos um clipe.")
    job_id = uuid.uuid4().hex
    with compilation_lock:
        compilation_jobs[job_id] = {"session_id": session_id, "status": "queued", "stage": "Na fila", "progress": 0.0, "cancel_requested": False}
    threading.Thread(target=run_compilation, args=(session_id, job_id, request.segment_ids), daemon=True).start()
    return {"job_id": job_id}


@app.get("/sessions/{session_id}/compilations/{job_id}")
def compilation_status(session_id: str, job_id: str) -> dict[str, object]:
    session_or_404(session_id)
    job = compilation_jobs.get(job_id)
    if not job or job.get("session_id") != session_id:
        raise HTTPException(404, "Compilação não encontrada.")
    return {key: value for key, value in job.items() if key not in {"process", "path", "cancel_requested", "session_id"}}


@app.delete("/sessions/{session_id}/compilations/{job_id}")
def cancel_compilation(session_id: str, job_id: str) -> dict[str, bool]:
    session_or_404(session_id)
    job = compilation_jobs.get(job_id)
    if not job or job.get("session_id") != session_id:
        raise HTTPException(404, "Compilação não encontrada.")
    if job.get("status") not in {"ready", "failed", "cancelled"}:
        job["cancel_requested"] = True
    return {"ok": True}


@app.get("/sessions/{session_id}/compilations/{job_id}/download")
def download_compilation(session_id: str, job_id: str) -> FileResponse:
    session_or_404(session_id)
    job = compilation_jobs.get(job_id)
    if not job or job.get("session_id") != session_id or job.get("status") != "ready":
        raise HTTPException(409, "A compilação ainda não está pronta.")
    return FileResponse(str(job["path"]), media_type="video/mp4", filename=str(job["filename"]))


@app.get("/sessions/{session_id}/source")
def source(session_id: str) -> FileResponse:
    path = Path(session_or_404(session_id)["video"])
    if not path.is_file():
        raise HTTPException(404, "Vídeo original não está disponível.")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")


@app.delete("/sessions/{session_id}")
def close(session_id: str) -> dict[str, bool]:
    state = session_or_404(session_id)
    if state.get("status") in {"running", "starting", "cancelling"}:
        raise HTTPException(409, "Não é possível remover uma análise enquanto ela está processando.")
    for job_id, job in list(compilation_jobs.items()):
        if job.get("session_id") == session_id:
            if job.get("status") not in {"ready", "failed", "cancelled"}:
                job["cancel_requested"] = True
            compilation_jobs.pop(job_id, None)
    shutil.rmtree(Path(state["workspace"]), ignore_errors=True)
    del sessions[session_id]
    return {"ok": True}
