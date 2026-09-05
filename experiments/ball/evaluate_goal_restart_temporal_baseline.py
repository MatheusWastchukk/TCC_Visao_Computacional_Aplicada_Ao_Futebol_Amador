"""Avalia reinício por sequência curta com validação leave-one-match-out."""
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


def image_features(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (192, 108), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values: list[float] = []
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    values.extend(cv2.normalize(histogram, histogram).flatten().tolist())
    edges = cv2.Canny(gray, 80, 160)
    for y in range(0, 108, 27):
        for x in range(0, 192, 48):
            cell = hsv[y : y + 27, x : x + 48]
            values.extend(cell.mean(axis=(0, 1)).tolist())
            values.append(float(edges[y : y + 27, x : x + 48].mean()))
    return np.asarray(values, dtype=np.float32)


def sequence_features(video_path: Path, timestamp: float, offsets: list[float]) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    values: list[np.ndarray] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        for offset in offsets:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((timestamp + offset) * fps)))
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Não foi possível ler {video_path} em {timestamp + offset:.3f}s.")
            values.append(image_features(image))
    finally:
        capture.release()
    stack = np.vstack(values)
    return np.concatenate((stack.mean(axis=0), stack.std(axis=0), np.abs(stack[-1] - stack[0])))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "samples": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_restart": float(precision), "recall_restart": float(recall), "f1_restart": float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia sequência visual curta para reconhecer reinício no centro.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-2, -1, 0, 1, 2])
    parser.add_argument("--external-only", action="store_true")
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["restart_label"] in LABEL_TO_TARGET and row["annotation_status"] == "completed"]
    if args.external_only:
        rows = [row for row in rows if row["sampling_group"] == "AMOSTRA_ALEATORIA_PARTIDA_SEPARADA"]
    x = np.vstack([sequence_features(args.videos_dir / row["video_filename"], float(row["sample_timestamp_seconds"]), args.offsets) for row in rows])
    y = np.asarray([LABEL_TO_TARGET[row["restart_label"]] for row in rows])
    groups = np.asarray([row["match_id"] for row in rows])
    probabilities = np.zeros(len(rows))
    for match in sorted(set(groups)):
        test = groups == match
        train = ~test
        if len(set(y[train])) != 2:
            raise ValueError(f"A retirada de {match} deixa o treino sem as duas classes.")
        scaler = StandardScaler()
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260902)
        model.fit(scaler.fit_transform(x[train]), y[train])
        probabilities[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    result_rows = []
    for row, target, probability, prediction in zip(rows, y, probabilities, predictions):
        result_rows.append({"sample_id": row["sample_id"], "match_id": row["match_id"], "sampling_group": row["sampling_group"], "true_label": row["restart_label"], "predicted_label": "REINICIO_CENTRO" if prediction else "NAO_REINICIO_CENTRO", "restart_probability": f"{probability:.6f}", "correct": str(bool(target == prediction))})
    with (args.output / "restart_temporal_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader(); writer.writerows(result_rows)
    summary = {"experiment_id": args.output.name, "evaluation_type": "leave_one_match_out", "external_only": args.external_only, "temporal_offsets_seconds": args.offsets, "feature_type": "media_desvio_e_delta_de_histograma_hsv_grade_espacial_bordas", "excluded_labels": ["INCONCLUSIVO"], **metrics(y, predictions), "warning": "Baseline temporal exploratório; dados ainda insuficientes para conclusão final."}
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
