"""Protege a seleção de CPU e a separação entre recursos e dados graváveis."""
import sys
from pathlib import Path

import pytest
import torch

from app import runtime


def test_cpu_default_even_with_available_gpu(monkeypatch):
    monkeypatch.delenv("GOAL_REVIEW_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runtime.processing_device() == "cpu"


def test_auto_falls_back_and_explicit_unavailable_gpu_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("GOAL_REVIEW_DEVICE", "auto")
    assert runtime.processing_device() == "cpu"
    monkeypatch.setenv("GOAL_REVIEW_DEVICE", "cuda:0")
    with pytest.raises(ValueError, match="Dispositivo indisponível"):
        runtime.processing_device()


def test_packaged_data_is_outside_read_only_resources(monkeypatch, tmp_path):
    resources = tmp_path / "installation"
    user_data = tmp_path / "user"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(resources), raising=False)
    monkeypatch.setenv("GOAL_REVIEW_DATA_DIR", str(user_data))
    monkeypatch.delenv("GOAL_REVIEW_FFMPEG", raising=False)
    monkeypatch.delenv("GOAL_REVIEW_RESNET_WEIGHTS", raising=False)
    assert runtime.resource_root() == resources
    assert runtime.session_root() == user_data / "sessions"
    assert runtime.ffmpeg_path() == resources / "ffmpeg" / "ffmpeg.exe"
    assert runtime.encoder_weights_path() == resources / "models" / "resnet18-f37072fd.pth"


def test_development_preserves_existing_session_directory(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("GOAL_REVIEW_DATA_DIR", raising=False)
    assert runtime.session_root() == Path(__file__).resolve().parents[1] / "outputs/review_app/sessions"
