"""Treina em todos os dados TRAIN os modelos já congelados do pipeline de gol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def main() -> int:
    parser = argparse.ArgumentParser(description="Materializa os modelos finais TRAIN do pipeline de eventos de gol.")
    parser.add_argument("--ranker-dataset", type=Path, required=True)
    parser.add_argument("--verifier-dataset", type=Path, required=True)
    parser.add_argument("--sequence-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    ranker_rows, ranker_fields = read_csv(args.ranker_dataset)
    feature_names = [field for field in ranker_fields if field not in {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}]
    ranker_x = np.array([[float(row[field]) for field in feature_names] for row in ranker_rows], dtype=np.float32)
    ranker_y = np.array([int(row["weak_label"]) for row in ranker_rows], dtype=int)
    ranker = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    ranker.fit(ranker_x, ranker_y)

    verifier_rows, _ = read_csv(args.verifier_dataset)
    if any(row["split"] != "train" for row in verifier_rows):
        raise AssertionError("O verificador final aceita somente TRAIN; VALIDATION e TEST permanecem fechados.")
    archive = np.load(args.sequence_embeddings, allow_pickle=False)
    embeddings_by_id = {clip_id: embedding for clip_id, embedding in zip(archive["clip_ids"], archive["embeddings"])}
    verifier_x = np.stack([embeddings_by_id[row["clip_id"]].reshape(-1) for row in verifier_rows])
    verifier_y = np.array([int(row["label"] == "GOAL") for row in verifier_rows], dtype=int)
    verifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
    verifier.fit(verifier_x, verifier_y)

    args.output.mkdir(parents=True)
    joblib.dump({"model": ranker, "features": feature_names}, args.output / "temporal_ranker.joblib")
    joblib.dump({"model": verifier, "frames_per_clip": 8, "embedding_dimensions": 512}, args.output / "temporal_sequence_verifier.joblib")
    metadata = {
        "experiment_id": args.output.name, "scope": "somente TRAIN", "ranker": {"groups": len(ranker_rows), "features": feature_names, "model": "RandomForest 300/min_leaf=2/balanced"},
        "verifier": {"clips": len(verifier_rows), "labels": {"GOAL": int(verifier_y.sum()), "NO_GOAL": int((~verifier_y.astype(bool)).sum())}, "model": "ResNet18 congelada + regressão logística L2"},
        "warning": "Modelos treinados em todos os dados TRAIN após seleção congelada; não são métricas de avaliação.",
    }
    (args.output / "metadata_v1.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
