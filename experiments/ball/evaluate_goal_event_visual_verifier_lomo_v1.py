"""Avalia um baseline visual de clipes de gol com separação por partida.

O experimento usa descritores visuais compactos dos oito frames de cada clipe.
Ele é um baseline de desenvolvimento: não usa VALIDATION nem TEST e não tenta
substituir uma avaliação independente do verificador final.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def frame_descriptor(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível ler o frame: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    scalars = np.array([
        gray.mean() / 255.0, gray.std() / 255.0,
        hsv[..., 1].mean() / 255.0, hsv[..., 1].std() / 255.0,
        hsv[..., 2].mean() / 255.0, (edges > 0).mean(),
    ], dtype=np.float32)
    hue_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    sat_hist = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
    histogram = np.concatenate([hue_hist / hue_hist.sum(), sat_hist / sat_hist.sum()])
    return np.concatenate([scalars, histogram.astype(np.float32)])


def clip_descriptor(frame_paths: list[Path]) -> np.ndarray:
    values = np.stack([frame_descriptor(path) for path in frame_paths])
    scalar_count = 6
    scalars, histograms = values[:, :scalar_count], values[:, scalar_count:]
    scalar_features = np.concatenate([
        scalars.mean(axis=0), scalars.std(axis=0),
        scalars[-1] - scalars[0],
        np.mean(np.abs(np.diff(scalars, axis=0)), axis=0),
    ])
    histogram_features = np.concatenate([
        histograms.mean(axis=0),
        np.mean(np.abs(np.diff(histograms, axis=0)), axis=0),
    ])
    return np.concatenate([scalar_features, histogram_features]).astype(np.float32)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    decisions = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, decisions, average="binary", zero_division=0
    )
    return {
        "examples": int(len(labels)), "positive_examples": int(labels.sum()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "precision_at_0_5": float(precision), "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1), "predicted_positive_at_0_5": int(decisions.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia baseline visual de clipes de gol por leave-one-match-out."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-role", default="candidate_review")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saida ja existe: {args.output}")

    manifest = [row for row in read_csv(args.manifest) if row["sampling_role"] == args.sampling_role]
    if not manifest:
        raise ValueError("Nenhum clipe corresponde ao sampling-role solicitado.")
    if any(row["split"] != "train" for row in manifest):
        raise AssertionError("O experimento aceita somente TRAIN; VALIDATION e TEST permanecem fechados.")
    indexed: dict[str, list[tuple[int, Path]]] = {}
    for row in read_csv(args.frame_index):
        indexed.setdefault(row["verifier_example_id"], []).append(
            (int(row["frame_order"]), Path(row["frame_path"]))
        )
    root = Path.cwd()
    features, labels, matches, example_ids = [], [], [], []
    for row in manifest:
        entries = sorted(indexed.get(row["verifier_example_id"], []))
        if len(entries) != 8:
            raise ValueError(f"{row['verifier_example_id']} deveria ter 8 frames, recebeu {len(entries)}.")
        frame_paths = [root / path for _, path in entries]
        features.append(clip_descriptor(frame_paths))
        labels.append(int(row["label"] == "GOAL"))
        matches.append(row["match_id"])
        example_ids.append(row["verifier_example_id"])
    x, y = np.stack(features), np.array(labels, dtype=int)
    groups = np.array(matches)
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    prediction_rows: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "experiment_id": args.output.name, "scope": "somente TRAIN",
        "protocol": "leave-one-match-out", "sampling_role": args.sampling_role,
        "examples": int(len(y)), "labels": dict(Counter("GOAL" if value else "NO_GOAL" for value in y)),
        "features": int(x.shape[1]), "models": {},
        "warning": "Baseline visual de desenvolvimento. Esta amostra não é conjunto independente e seus resultados não estimam desempenho final.",
    }
    for model_name, model in models.items():
        scores = np.zeros(len(y), dtype=float)
        for held_match in sorted(set(groups)):
            test_mask = groups == held_match
            train_mask = ~test_mask
            if len(np.unique(y[train_mask])) < 2:
                raise ValueError(f"Treino sem ambas as classes ao separar {held_match}.")
            model.fit(x[train_mask], y[train_mask])
            scores[test_mask] = model.predict_proba(x[test_mask])[:, 1]
        summary["models"][model_name] = metrics(y, scores)
        for idx, score in enumerate(scores):
            prediction_rows.append({
                "model": model_name, "verifier_example_id": example_ids[idx], "match_id": matches[idx],
                "label": "GOAL" if y[idx] else "NO_GOAL", "score_goal": f"{score:.8f}",
                "prediction_at_0_5": "GOAL" if score >= 0.5 else "NO_GOAL",
            })
    args.output.mkdir(parents=True)
    with (args.output / "visual_features_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["verifier_example_id", "match_id", "label"] + [f"feature_{i:03d}" for i in range(x.shape[1])])
        for example_id, match, label, vector in zip(example_ids, matches, y, x):
            writer.writerow([example_id, match, "GOAL" if label else "NO_GOAL"] + [f"{value:.8f}" for value in vector])
    with (args.output / "predictions_lomo_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
