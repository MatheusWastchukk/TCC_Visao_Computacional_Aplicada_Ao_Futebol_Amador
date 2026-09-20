"""Pontua visualmente e combina candidatos sem consultar rótulos de referência."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def read_clip_frames(video_path: Path, start: float, end: float, count: int = 8) -> list[Image.Image]:
    """Lê amostras equidistantes de um intervalo de vídeo."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
    frames: list[Image.Image] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise RuntimeError(f"Não foi possível obter a duração de {video_path}.")
        last_timestamp = max(0.0, (frame_count - 2) / fps)
        bounded_start = min(max(0.0, start), last_timestamp)
        bounded_end = min(max(bounded_start + 1.0 / fps, end), last_timestamp)
        for timestamp in np.linspace(bounded_start, bounded_end, count, endpoint=False):
            target = min(float(timestamp), last_timestamp)
            capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                # Alguns MP4s longos falham no seek por milissegundo perto de keyframes.
                capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 2, max(0, round(target * fps))))
                ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Não foi possível ler {video_path} em {target:.3f}s após tentativas de seek.")
            frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return frames


def visual_scores(candidates: Iterable[dict[str, str]], videos_dir: Path, verifier_path: Path, torch_threads: int = 4) -> list[dict[str, str]]:
    """Aplica o verificador congelado como escore de reordenação, nunca como decisão binária."""
    rows = [dict(row) for row in candidates]
    frozen = joblib.load(verifier_path)
    if frozen.get("frames_per_clip") != 8 or frozen.get("embedding_dimensions") != 512:
        raise ValueError("Contrato incompatível do verificador visual congelado.")
    torch.set_num_threads(torch_threads)
    transform = ResNet18_Weights.DEFAULT.transforms()
    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    batches: list[torch.Tensor] = []
    embeddings: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        frames = read_clip_frames(videos_dir / row["video_filename"], float(row["candidate_start_seconds"]), float(row["candidate_end_seconds"]))
        batches.append(torch.stack([transform(frame) for frame in frames]))
        if len(batches) == 8 or index == len(rows):
            with torch.inference_mode():
                encoded = encoder(torch.cat(batches)).cpu().numpy().astype(np.float32)
            embeddings.extend(encoded[offset * 8:(offset + 1) * 8].reshape(-1) for offset in range(len(batches)))
            batches.clear()
    scores = frozen["model"].predict_proba(np.stack(embeddings))[:, 1]
    return [{**row, "visual_score_goal": f"{float(score):.8f}"} for row, score in zip(rows, scores)]


def fusion_scores(candidates: Iterable[dict[str, str]], fusion_path: Path) -> list[dict[str, str]]:
    """Aplica a fusão logística congelada sem recalibração ou métricas de avaliação."""
    rows = [dict(row) for row in candidates]
    frozen = joblib.load(fusion_path)
    if frozen.get("features") != ["rank_score", "visual_score_goal"]:
        raise ValueError("Contrato incompatível do modelo de fusão congelado.")
    matrix = np.asarray([[float(row["rank_score"]), float(row["visual_score_goal"])] for row in rows], dtype=np.float64)
    scores = frozen["model"].predict_proba(matrix)[:, 1]
    return [{**row, "fusion_score_goal": f"{float(score):.8f}"} for row, score in zip(rows, scores)]
