"""Verifica a inferência visual com pesos locais e sem download implícito."""
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

from src.events import visual_fusion_inference_v4 as visual


def test_visual_scores_loads_local_encoder_without_pretrained_download(monkeypatch, tmp_path):
    weights = tmp_path / "encoder.pth"
    model = resnet18(weights=None)
    torch.save(model.state_dict(), weights)
    observed = []

    def local_encoder(*, weights):
        # Qualquer solicitação de pesos automáticos invalidaria o teste offline.
        assert weights is None
        return resnet18(weights=None)

    class Classifier:
        def predict_proba(self, matrix):
            observed.append(matrix)
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(visual, "resnet18", local_encoder)
    monkeypatch.setattr(visual.joblib, "load", lambda _: {
        "frames_per_clip": 8, "embedding_dimensions": 512, "model": Classifier(),
    })
    monkeypatch.setattr(visual, "read_clip_frames", lambda *_: [Image.new("RGB", (224, 224))] * 8)
    rows = visual.visual_scores([
        {"video_filename": "demo.mp4", "candidate_start_seconds": "0", "candidate_end_seconds": "5"},
    ], tmp_path, tmp_path / "verifier.joblib", encoder_weights=weights)
    assert observed[0].shape == (1, 4096)
    assert np.isfinite(observed[0]).all()
    assert rows[0]["visual_score_goal"] == "0.80000000"
