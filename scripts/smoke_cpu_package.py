"""Exercita a API empacotada com vídeo autorizado, sem consultar rótulos de TEST."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8765"


def request(route: str, method: str = "GET", body: dict | None = None, raw: bool = False):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(BASE + route, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        content = response.read()
    return content if raw else json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa análise, exportação e restauração no backend escolhido.")
    parser.add_argument("--backend", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path, help="Vídeo TRAIN ou externo autorizado; não usar TEST fechado.")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--mode", choices=("cpu", "nvidia", "amd"), default="cpu")
    args = parser.parse_args()
    backend, video, workspace = args.backend.absolute(), args.video.resolve(), args.workspace.resolve()
    if not backend.is_file() or not video.is_file():
        raise FileNotFoundError("Backend e vídeo devem existir.")
    try:
        request("/health")
    except URLError:
        pass
    else:
        raise RuntimeError("A porta 8765 já está ocupada. Encerre a outra instância antes do teste.")
    workspace.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "GOAL_REVIEW_DEVICE": "cpu" if args.mode == "cpu" else "cuda:0",
           "GOAL_REVIEW_ACCELERATOR": args.mode, "GOAL_REVIEW_DATA_DIR": str(workspace / "data"), "PYTHONUTF8": "1",
           "MIOPEN_USER_DB_PATH": str(workspace / "miopen-db"), "MIOPEN_CUSTOM_CACHE_DIR": str(workspace / "miopen-kernels")}
    for key in ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    log = (workspace / "backend.log").open("w", encoding="utf-8")
    process = None

    def start_backend():
        child = subprocess.Popen([str(backend)], cwd=workspace, env=env, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"Backend encerrou com código {child.returncode}; consulte backend.log.")
                try:
                    request("/health")
                    return child
                except URLError:
                    time.sleep(0.5)
            raise TimeoutError("A API não iniciou em 120 segundos.")
        except BaseException:
            child.terminate()
            child.wait(timeout=15)
            raise

    def stop_backend(child):
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    try:
        process = start_backend()
        session_id = request("/sessions", "POST", {"video_path": str(video), "mode": "cobertura_reduzida"})["session_id"]
        route = f"/sessions/{session_id}"
        deadline = time.monotonic() + args.timeout
        last_stage = None
        cpu_samples = []
        while time.monotonic() < deadline:
            state = request(route)
            cpu = state.get("metrics", {}).get("process_cpu_percent")
            if cpu is not None:
                cpu_samples.append(cpu)
            if state.get("stage") != last_stage:
                last_stage = state.get("stage")
                print(last_stage, flush=True)
            if state["status"] == "completed":
                break
            if state["status"] in {"failed", "cancelled", "interrupted"}:
                raise RuntimeError(f"Análise não concluída: {state.get('error')}")
            time.sleep(1)
        else:
            raise TimeoutError("A análise excedeu o limite do teste.")
        clips = request(route + "/clips")
        if not any(sample > 0 for sample in cpu_samples):
            raise RuntimeError("O indicador de CPU não registrou atividade durante a análise.")
        expected = "CPU" if args.mode == "cpu" else "GPU"
        if not clips or state["metrics"]["processor"] != expected or state["processing_backend"] != args.mode:
            raise RuntimeError("O teste exige clipes gerados no modo de processamento escolhido.")
        first_id = clips[0]["segment_id"]
        media = request(route + f"/clips/{first_id}/media", raw=True)
        if b"ftyp" not in media[:32]:
            raise RuntimeError("O clipe retornado não é MP4.")
        note = f"Teste técnico {args.mode}; sem anotação de referência."
        request(route + f"/clips/{first_id}", "PATCH", {"notes": note})
        for extension in ("csv", "json"):
            (workspace / f"export.{extension}").write_bytes(request(route + f"/export?format={extension}", raw=True))
        (workspace / "report.html").write_bytes(request(route + "/report.html", raw=True))
        stop_backend(process)
        process = None
        process = start_backend()
        restored = request(route)
        restored_clips = request(route + "/clips")
        if restored["status"] != "completed" or len(restored_clips) != len(clips):
            raise RuntimeError("A restauração da sessão não preservou os resultados.")
        if next(item for item in restored_clips if item["segment_id"] == first_id).get("notes") != note:
            raise RuntimeError("A nota não foi preservada após reiniciar o backend.")
        result = {
            "status": "ok", "scope": "smoke técnico em vídeo autorizado, sem avaliação TEST",
            "video": str(video), "backend": str(backend), "session_id": session_id,
            "analysis_seconds": state["elapsed"], "clips": len(clips),
            "device": state["metrics"]["processor"], "media_bytes": len(media),
            "processing_backend": state["processing_backend"],
            "exports": ["csv", "json", "html"], "session_restored": True,
            "max_backend_cpu_percent": max(cpu_samples),
            "stage_seconds": state.get("stage_seconds", {}),
        }
        (workspace / "smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if process is not None:
            stop_backend(process)
        log.close()


if __name__ == "__main__":
    main()
