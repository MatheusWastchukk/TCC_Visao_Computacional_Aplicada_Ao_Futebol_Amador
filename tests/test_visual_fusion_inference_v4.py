from src.events.visual_fusion_inference_v4 import fusion_scores
import numpy as np


class Modelo:
    def predict_proba(self, matrix):
        return np.asarray([[1.0 - row[0], row[0]] for row in matrix])


def test_fusion_scores_uses_frozen_contract(monkeypatch) -> None:
    monkeypatch.setattr("src.events.visual_fusion_inference_v4.joblib.load", lambda _: {"features": ["rank_score", "visual_score_goal"], "model": Modelo()})
    rows = fusion_scores([{"rank_score": "0.7", "visual_score_goal": "0.2"}], "modelo.joblib")
    assert rows[0]["fusion_score_goal"] == "0.70000000"
