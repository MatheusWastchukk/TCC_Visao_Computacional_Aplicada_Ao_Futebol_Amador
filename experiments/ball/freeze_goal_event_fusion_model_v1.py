"""Materializa a fusão final a partir de escores out-of-fold de TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina a fusão final somente com escores out-of-fold de TRAIN.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    with args.candidate_scores.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"match_id", "rank_score", "visual_score_goal", "catalog_goal_in_clip"}
    missing = required - set(rows[0]) if rows else required
    if not rows or missing:
        raise ValueError(f"Escores sem colunas: {sorted(missing)}")
    x = np.asarray([[float(row["rank_score"]), float(row["visual_score_goal"])] for row in rows], dtype=np.float64)
    y = np.asarray([int(row["catalog_goal_in_clip"]) for row in rows], dtype=int)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
    model.fit(x, y)
    args.output.mkdir(parents=True)
    joblib.dump({"model": model, "features": ["rank_score", "visual_score_goal"]}, args.output / "fusion_reranker.joblib")
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN; escores de entrada out-of-fold",
        "candidates": len(rows), "positive_candidates": int(y.sum()), "matches": len({row["match_id"] for row in rows}),
        "model": "regressão logística L2 balanceada", "features": ["rank_score", "visual_score_goal"],
        "warning": "Modelo final para confirmação única; não é métrica de avaliação.",
    }
    (args.output / "metadata_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
