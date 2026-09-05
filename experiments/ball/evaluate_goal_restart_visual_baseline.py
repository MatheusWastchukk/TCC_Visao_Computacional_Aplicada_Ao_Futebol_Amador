"""Executa baseline visual de reinício com validação leave-one-match-out."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


LABEL_TO_TARGET = {"NAO_REINICIO_CENTRO": 0, "REINICIO_CENTRO": 1}


def extract_features(video_path: Path, timestamp: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Não foi possível ler {video_path} em {timestamp:.3f}s.")
    image = cv2.resize(image, (192, 108), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features: list[float] = []
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    features.extend(histogram.tolist())
    edges = cv2.Canny(gray, 80, 160)
    for y in range(0, 108, 27):
        for x in range(0, 192, 48):
            cell_hsv = hsv[y : y + 27, x : x + 48]
            features.extend(cell_hsv.mean(axis=(0, 1)).tolist())
            features.append(float(edges[y : y + 27, x : x + 48].mean()))
    return np.asarray(features, dtype=np.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "samples": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_restart": float(precision), "recall_restart": float(recall), "f1_restart": float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia baseline visual de reinício, deixando uma partida inteira fora por vez."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Usa somente as amostras aleatórias de partidas separadas.",
    )
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["annotation_status"] == "completed" and row["restart_label"] in LABEL_TO_TARGET
        ]
    if args.external_only:
        rows = [row for row in rows if row["sampling_group"] == "AMOSTRA_ALEATORIA_PARTIDA_SEPARADA"]
    if len({row["match_id"] for row in rows}) < 2 or len({row["restart_label"] for row in rows}) != 2:
        raise ValueError("São necessárias ao menos duas partidas e as duas classes para a avaliação.")

    features = np.vstack([
        extract_features(args.videos_dir / row["video_filename"], float(row["sample_timestamp_seconds"]))
        for row in rows
    ])
    targets = np.asarray([LABEL_TO_TARGET[row["restart_label"]] for row in rows], dtype=np.int64)
    matches = np.asarray([row["match_id"] for row in rows])
    probabilities = np.zeros(len(rows), dtype=np.float64)
    predictions = np.zeros(len(rows), dtype=np.int64)
    for match in sorted(set(matches)):
        test_mask = matches == match
        train_mask = ~test_mask
        if len(set(targets[train_mask])) != 2:
            raise ValueError(f"A retirada de {match} deixa o treino sem as duas classes.")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_mask])
        x_test = scaler.transform(features[test_mask])
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260902)
        model.fit(x_train, targets[train_mask])
        probabilities[test_mask] = model.predict_proba(x_test)[:, 1]
        predictions[test_mask] = (probabilities[test_mask] >= 0.5).astype(np.int64)

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for row, target, probability, prediction in zip(rows, targets, probabilities, predictions):
        prediction_rows.append(
            {
                "sample_id": row["sample_id"], "match_id": row["match_id"],
                "sampling_group": row["sampling_group"], "true_label": row["restart_label"],
                "predicted_label": "REINICIO_CENTRO" if prediction else "NAO_REINICIO_CENTRO",
                "restart_probability": f"{probability:.6f}", "correct": str(bool(target == prediction)),
            }
        )
    with (args.output / "restart_visual_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    metrics = {
        "experiment_id": args.output.name,
        "evaluation_type": "leave_one_match_out",
        "external_only": args.external_only,
        "feature_type": "histograma_hsv_e_grade_espacial_hsv_bordas",
        "excluded_labels": ["INCONCLUSIVO"],
        **metric_dict(targets, predictions),
        "matches": sorted(set(matches)),
        "warning": "Baseline exploratório com poucos reinícios; não representa detector final nem validação fechada.",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
