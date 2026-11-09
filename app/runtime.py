"""Resolve recursos de leitura e dados graváveis no desenvolvimento e no pacote."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_root() -> Path:
    """Retorna a raiz dos recursos incluídos pelo PyInstaller ou do checkout."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def session_root() -> Path:
    """Preserva sessões locais; no pacote, usa uma pasta gravável do usuário."""
    configured = os.environ.get("GOAL_REVIEW_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve() / "sessions"
    if getattr(sys, "frozen", False):
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return local / "GoalEventReview" / "sessions"
    return resource_root() / "outputs" / "review_app" / "sessions"


def ffmpeg_path() -> Path:
    """Localiza o FFmpeg sem depender do PATH ou de npm na máquina de destino."""
    configured = os.environ.get("GOAL_REVIEW_FFMPEG")
    if configured:
        return Path(configured).expanduser().resolve()
    root = resource_root()
    if getattr(sys, "frozen", False):
        return root / "ffmpeg" / "ffmpeg.exe"
    return root / "desktop" / "node_modules" / "ffmpeg-static" / "ffmpeg.exe"


def processing_device() -> str:
    """Usa CPU por padrão; GPU continua disponível por configuração explícita."""
    import torch

    requested = os.environ.get("GOAL_REVIEW_DEVICE", "cpu").strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "cuda:0" and torch.cuda.is_available():
        return requested
    raise ValueError(
        "Dispositivo indisponível ou inválido em GOAL_REVIEW_DEVICE. "
        "Use cpu, auto ou cuda:0 com GPU compatível."
    )


def encoder_weights_path() -> Path | None:
    """No pacote, exige pesos locais para impedir download durante a análise."""
    configured = os.environ.get("GOAL_REVIEW_RESNET_WEIGHTS")
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return resource_root() / "models" / "resnet18-f37072fd.pth"
    return None
