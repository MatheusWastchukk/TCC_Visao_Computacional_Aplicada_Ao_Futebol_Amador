"""Avalia embeddings congelados ResNet18 para clipes de gol, somente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_clip_frames(video_path: Path, start: float, end: float, count: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo TRAIN: {video_path}")
    frames: list[Image.Image] = []
    try:
        for timestamp in np.linspace(start, end, count, endpoint=False):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Não foi possível ler {video_path} em {timestamp:.3f}s.")
            frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return frames


def clip_embedding(
    video_path: Path, start: float, end: float, frame_count: int, transform: object, model: torch.nn.Module
) -> np.ndarray:
    frames = read_clip_frames(video_path, start, end, frame_count)
    batch = torch.stack([transform(frame) for frame in frames])
    with torch.inference_mode():
        embeddings = model(batch).cpu().numpy()
    return np.concatenate([
        embeddings.mean(axis=0), embeddings.std(axis=0),
        embeddings[-1] - embeddings[0], np.abs(np.diff(embeddings, axis=0)).mean(axis=0),
    ]).astype(np.float32)


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    decisions = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, decisions, average="binary", zero_division=0)
    return {
        "examples": int(len(labels)), "positive_examples": int(labels.sum()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "precision_at_0_5": float(precision), "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1), "predicted_positive_at_0_5": int(decisions.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia ResNet18 congelada em clipes TRAIN por leave-one-match-out.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.frames_per_clip < 2:
        raise ValueError("São necessários ao menos dois frames por clipe.")
    torch.set_num_threads(args.torch_threads)
    rows = read_csv(args.dataset)
    if not rows or any(row["split"] != "train" for row in rows):
        raise AssertionError("O experimento aceita somente TRAIN; VALIDATION e TEST permanecem fechados.")
    if {row["label"] for row in rows} != {"GOAL", "NO_GOAL"}:
        raise ValueError("O dataset precisa conter as classes GOAL e NO_GOAL.")
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    features, labels, matches, clip_ids, sources = [], [], [], [], []
    for index, row in enumerate(rows, start=1):
        features.append(clip_embedding(
            args.videos_dir / row["video_filename"], float(row["clip_start_seconds"]),
            float(row["clip_end_seconds"]), args.frames_per_clip, transform, model,
        ))
        labels.append(int(row["label"] == "GOAL")); matches.append(row["match_id"])
        clip_ids.append(row["clip_id"]); sources.append(row["source_type"])
        if index % 25 == 0 or index == len(rows):
            print(f"Embeddings extraídos: {index}/{len(rows)}")
    x, y, groups = np.stack(features), np.array(labels, dtype=int), np.array(matches)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    predictions: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "experiment_id": args.output.name, "scope": "somente TRAIN", "protocol": "leave-one-match-out",
        "encoder": "ResNet18 ImageNet congelada", "frames_per_clip": args.frames_per_clip,
        "examples": len(rows), "labels": dict(Counter(row["label"] for row in rows)),
        "matches": len(set(matches)), "features": int(x.shape[1]), "models": {},
        "warning": "Desenvolvimento em TRAIN. O conjunto usa negativos temporais com auditoria amostral e não é avaliação final independente.",
    }
    for model_name, classifier in models.items():
        scores = np.zeros(len(y), dtype=float)
        for held_match in sorted(set(groups)):
            test_mask, train_mask = groups == held_match, groups != held_match
            if len(np.unique(y[train_mask])) < 2:
                raise ValueError(f"Treino sem ambas as classes ao separar {held_match}.")
            classifier.fit(x[train_mask], y[train_mask])
            scores[test_mask] = classifier.predict_proba(x[test_mask])[:, 1]
        summary["models"][model_name] = classification_metrics(y, scores)
        for idx, score in enumerate(scores):
            predictions.append({
                "model": model_name, "clip_id": clip_ids[idx], "match_id": matches[idx], "label": "GOAL" if y[idx] else "NO_GOAL",
                "source_type": sources[idx], "score_goal": f"{score:.8f}", "prediction_at_0_5": "GOAL" if score >= 0.5 else "NO_GOAL",
            })
    args.output.mkdir(parents=True)
    with (args.output / "clip_embeddings_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["clip_id", "match_id", "label", "source_type"] + [f"feature_{index:04d}" for index in range(x.shape[1])])
        for row, vector in zip(rows, x):
            writer.writerow([row["clip_id"], row["match_id"], row["label"], row["source_type"]] + [f"{value:.8f}" for value in vector])
    with (args.output / "predictions_lomo_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
