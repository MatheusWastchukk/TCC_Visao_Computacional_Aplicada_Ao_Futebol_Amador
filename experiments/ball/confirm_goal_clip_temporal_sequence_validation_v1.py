"""Confirma uma única configuração de sequência congelada em VALIDATION."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

from evaluate_goal_clip_pretrained_verifier_lomo_v1 import read_clip_frames


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def classification(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    return {"threshold": float(threshold), "tp": int(np.sum(predicted & (labels == 1))),
            "fp": int(np.sum(predicted & (labels == 0))), "tn": int(np.sum(~predicted & (labels == 0))),
            "fn": int(np.sum(~predicted & (labels == 1))), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "false_positive_rate": float(np.sum(predicted & (labels == 0)) / max(np.sum(labels == 0), 1))}


def sequence_features(video: Path, start: float, end: float, transform: object, encoder: torch.nn.Module) -> np.ndarray:
    images: list[Image.Image] = read_clip_frames(video, start, end, 8)
    batch = torch.stack([transform(frame) for frame in images])
    with torch.inference_mode():
        return encoder(batch).cpu().numpy().astype(np.float32).reshape(-1)


def aggregate(sequence: np.ndarray) -> np.ndarray:
    embeddings = sequence.reshape(8, 512)
    return np.concatenate([embeddings.mean(axis=0), embeddings.std(axis=0),
                           embeddings[-1] - embeddings[0], np.abs(np.diff(embeddings, axis=0)).mean(axis=0)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirma em VALIDATION um ranker de sequência com protocolo TRAIN congelado.")
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-sequence-embeddings", type=Path, required=True)
    parser.add_argument("--train-aggregate-embeddings", type=Path, required=True)
    parser.add_argument("--validation-candidates", type=Path, required=True)
    parser.add_argument("--historical-validation-scores", type=Path, help="Opcional: audita exposição anterior no mesmo conjunto VALIDATION.")
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-threshold", type=float, default=0.06779222)
    parser.add_argument("--aggregate-threshold", type=float, default=0.45027083)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")

    train_rows, _ = read_csv(args.train_dataset)
    valid_rows, _ = read_csv(args.validation_candidates)
    if any(row["split"] != "train" for row in train_rows) or any(row["split"] != "validation" for row in valid_rows):
        raise AssertionError("Os arquivos não correspondem aos splits TRAIN e VALIDATION esperados.")
    artifact = np.load(args.train_sequence_embeddings, allow_pickle=False)
    sequence_ids = artifact["clip_ids"].astype(str).tolist()
    train_ids = [row["clip_id"] for row in train_rows]
    if sequence_ids != train_ids:
        raise ValueError("IDs/ordem dos embeddings de sequência diferem do dataset TRAIN.")
    train_sequence = artifact["embeddings"].reshape(len(train_rows), -1)
    train_aggregate_rows, _ = read_csv(args.train_aggregate_embeddings)
    if [row["clip_id"] for row in train_aggregate_rows] != train_ids:
        raise ValueError("IDs/ordem dos embeddings agregados diferem do dataset TRAIN.")
    aggregate_fields = sorted(field for field in train_aggregate_rows[0] if field.startswith("feature_"))
    train_aggregate = np.asarray([[float(row[field]) for field in aggregate_fields] for row in train_aggregate_rows], dtype=np.float32)
    y_train = np.asarray([row["label"] == "GOAL" for row in train_rows], dtype=int)

    torch.set_num_threads(args.torch_threads)
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    valid_sequence = []
    for index, row in enumerate(valid_rows, 1):
        valid_sequence.append(sequence_features(args.videos_dir / row["video_filename"],
                                                float(row["candidate_start_seconds"]),
                                                float(row["candidate_end_seconds"]), transform, encoder))
        if index % 8 == 0 or index == len(valid_rows):
            print(f"Sequências VALIDATION extraídas: {index}/{len(valid_rows)}", flush=True)
    x_valid_sequence = np.stack(valid_sequence)
    x_valid_aggregate = np.stack([aggregate(item) for item in x_valid_sequence])
    y_valid = np.asarray([row["catalog_goal_in_clip"].lower() == "true" for row in valid_rows], dtype=int)

    sequence_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
    sequence_model.fit(train_sequence, y_train)
    sequence_scores = sequence_model.predict_proba(x_valid_sequence)[:, 1]
    aggregate_model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    aggregate_model.fit(train_aggregate, y_train)
    aggregate_scores = aggregate_model.predict_proba(x_valid_aggregate)[:, 1]

    predictions = []
    for index, row in enumerate(valid_rows):
        predictions.append({"event_id": row["event_id"], "match_id": row["match_id"],
                            "catalog_goal_in_clip": int(y_valid[index]),
                            "sequence_score": f"{sequence_scores[index]:.8f}",
                            "aggregate_rf_score": f"{aggregate_scores[index]:.8f}"})
    historical_audit = None
    if args.historical_validation_scores:
        historical_rows, _ = read_csv(args.historical_validation_scores)
        historical = {row["event_id"]: row["visual_score_goal"] for row in historical_rows}
        matches = [row for row in predictions if row["event_id"] in historical]
        exact = sum(row["sequence_score"] == historical[row["event_id"]] for row in matches)
        historical_audit = {"source": str(args.historical_validation_scores),
                            "matched_candidates": len(matches), "exact_score_matches": exact,
                            "interpretation": "Escores previamente observados no mesmo VALIDATION não contam como métrica independente nova."}
    metrics = {}
    for name, scores, threshold in [("temporal_sequence_logistic", sequence_scores, args.sequence_threshold),
                                    ("aggregate_random_forest", aggregate_scores, args.aggregate_threshold)]:
        metrics[name] = {"examples": int(len(y_valid)), "positive_examples": int(y_valid.sum()),
                         "average_precision": float(average_precision_score(y_valid, scores)),
                         "roc_auc": float(roc_auc_score(y_valid, scores)) if len(np.unique(y_valid)) == 2 else None,
                         "frozen_lomo_threshold_metrics": classification(y_valid, scores, threshold)}

    args.output.mkdir(parents=True)
    with (args.output / "validation_predictions_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    summary = {
        "experiment_id": args.output.name,
        "scope": "VALIDATION; pesos e limiares treinados/congelados em TRAIN; TEST não foi acessado",
        "train_examples": int(len(y_train)), "validation_candidates": int(len(y_valid)),
        "validation_positive_candidates": int(y_valid.sum()),
        "configuration": {"encoder": "ResNet18 ImageNet congelada", "frames_per_clip": 8,
                           "sequence_classifier": "LogisticRegression class_weight=balanced, max_iter=3000, random_state=42",
                           "aggregate_baseline": "RandomForestClassifier n_estimators=400, min_samples_leaf=2, class_weight=balanced, random_state=42",
                           "thresholds_frozen_from_train_lomo": {"sequence": args.sequence_threshold, "aggregate": args.aggregate_threshold}},
        "metrics": metrics,
        "historical_score_reproduction_audit": historical_audit,
        "warning": "A amostra contém somente 32 candidatos e pode ter exposição histórica; verifique escores VALIDATION prévios antes de alegar confirmação independente. Não ajustar com estes resultados nem extrapolar para TEST.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
