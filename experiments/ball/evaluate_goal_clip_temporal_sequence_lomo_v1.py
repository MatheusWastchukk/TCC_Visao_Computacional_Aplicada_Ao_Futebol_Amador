"""Avalia sequência ordenada de embeddings ResNet18 em clipes TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

from evaluate_goal_clip_pretrained_verifier_lomo_v1 import read_clip_frames, read_csv


def sequence_embedding(video_path: Path, start: float, end: float, count: int, transform: object, model: torch.nn.Module) -> np.ndarray:
    frames = read_clip_frames(video_path, start, end, count)
    batch = torch.stack([transform(frame) for frame in frames])
    with torch.inference_mode():
        return model(batch).cpu().numpy().astype(np.float32)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    predicted = scores >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    return {
        "examples": int(len(labels)), "positive_examples": int(labels.sum()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "precision_at_0_5": float(precision), "recall_at_0_5": float(recall), "f1_at_0_5": float(f1),
        "predicted_positive_at_0_5": int(predicted.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia sequência ResNet18 congelada por leave-one-match-out, somente TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.frames_per_clip != 8:
        raise ValueError("A versão V1 foi congelada para oito frames por clipe.")
    torch.set_num_threads(args.torch_threads)
    rows = read_csv(args.dataset)
    if not rows or any(row["split"] != "train" for row in rows):
        raise AssertionError("O experimento aceita somente TRAIN; VALIDATION e TEST permanecem fechados.")
    if {row["label"] for row in rows} != {"GOAL", "NO_GOAL"}:
        raise ValueError("O dataset precisa conter GOAL e NO_GOAL.")
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    encoder = resnet18(weights=weights); encoder.fc = torch.nn.Identity(); encoder.eval()
    sequences, labels, matches, clip_ids = [], [], [], []
    for index, row in enumerate(rows, start=1):
        sequences.append(sequence_embedding(
            args.videos_dir / row["video_filename"], float(row["clip_start_seconds"]),
            float(row["clip_end_seconds"]), args.frames_per_clip, transform, encoder,
        ))
        labels.append(int(row["label"] == "GOAL")); matches.append(row["match_id"]); clip_ids.append(row["clip_id"])
        if index % 25 == 0 or index == len(rows):
            print(f"Sequências extraídas: {index}/{len(rows)}")
    sequences_array = np.stack(sequences)
    x, y, groups = sequences_array.reshape(len(rows), -1), np.array(labels, dtype=int), np.array(matches)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
    scores = np.zeros(len(y), dtype=float)
    for held_match in sorted(set(groups)):
        test_mask, train_mask = groups == held_match, groups != held_match
        classifier.fit(x[train_mask], y[train_mask])
        scores[test_mask] = classifier.predict_proba(x[test_mask])[:, 1]
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "sequence_embeddings_v1.npz", clip_ids=np.array(clip_ids), embeddings=sequences_array)
    with (args.output / "predictions_lomo_v1.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["clip_id", "match_id", "label", "score_goal", "prediction_at_0_5"]
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for clip_id, match_id, label, score in zip(clip_ids, matches, y, scores):
            writer.writerow({"clip_id": clip_id, "match_id": match_id, "label": "GOAL" if label else "NO_GOAL", "score_goal": f"{score:.8f}", "prediction_at_0_5": "GOAL" if score >= 0.5 else "NO_GOAL"})
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN", "protocol": "leave-one-match-out",
        "hypothesis": "A ordem dos oito embeddings congelados contém informação adicional ao agregado estatístico.",
        "encoder": "ResNet18 ImageNet congelada", "classifier": "regressão logística L2 balanceada",
        "frames_per_clip": args.frames_per_clip, "sequence_features": int(x.shape[1]),
        "examples": len(rows), "labels": dict(Counter(row["label"] for row in rows)), "metrics": metrics(y, scores),
        "promotion_rule": "AP deve superar 66,68% por pelo menos 3 pontos percentuais e reduzir o compromisso recall-FP do baseline agregado.",
        "warning": "Desenvolvimento em TRAIN; não usar este resultado como avaliação final independente.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
