"""Extrai sinais simples de áudio de clipes e avalia LOMO exclusivamente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["rms_mean", "rms_p95", "rms_max", "rms_std", "silence_fraction", "zcr_mean",
            "spectral_centroid_mean", "spectral_flux_mean", "spectral_flux_p95", "low_high_energy_ratio"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para salvar: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def decode_audio_clip(path: Path, start: float, end: float, target_rate: int = 8000) -> np.ndarray:
    container = av.open(str(path))
    try:
        streams = list(container.streams.audio)
        if not streams:
            raise ValueError(f"Vídeo TRAIN sem faixa de áudio: {path.name}")
        stream = streams[0]
        container.seek(max(0, int(start / float(stream.time_base))), stream=stream, backward=True)
        pieces: list[np.ndarray] = []
        source_rate = int(stream.rate or stream.codec_context.sample_rate)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_start = float(frame.pts * frame.time_base)
            samples = frame.to_ndarray().astype(np.float32)
            mono = samples.mean(axis=0) if samples.ndim == 2 else samples.reshape(-1)
            frame_end = frame_start + len(mono) / int(frame.sample_rate or source_rate)
            if frame_start >= end:
                break
            if frame_end <= start:
                continue
            rate = int(frame.sample_rate or source_rate)
            first = max(0, int((start - frame_start) * rate))
            last = min(len(mono), int(np.ceil((end - frame_start) * rate)))
            if last > first:
                pieces.append(mono[first:last])
        if not pieces:
            raise RuntimeError(f"Áudio vazio no trecho {start:.2f}-{end:.2f}s de {path.name}.")
        signal = np.concatenate(pieces)
        if source_rate != target_rate:
            divisor = int(np.gcd(source_rate, target_rate))
            signal = resample_poly(signal, target_rate // divisor, source_rate // divisor).astype(np.float32)
        return signal
    finally:
        container.close()


def audio_features(signal: np.ndarray, sample_rate: int = 8000) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float32)
    block = sample_rate // 10
    usable = signal[:len(signal) // block * block]
    if not len(usable):
        raise ValueError("O sinal de áudio é curto demais para extrair atributos.")
    blocks = usable.reshape(-1, block)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-12)
    zcr = np.mean((blocks[:, 1:] * blocks[:, :-1]) < 0, axis=1)

    frame_size, hop = 1024, 512
    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)))
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_size), axis=1))
    normalized = spectrum / (spectrum.sum(axis=1, keepdims=True) + 1e-8)
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    centroid = normalized @ frequencies
    flux = np.maximum(np.diff(normalized, axis=0), 0).sum(axis=1)
    if not len(flux):
        flux = np.zeros(1, dtype=np.float32)
    power = spectrum.mean(axis=0) ** 2
    low = float(power[(frequencies >= 80) & (frequencies < 1000)].sum())
    high = float(power[(frequencies >= 1000) & (frequencies < 4000)].sum())
    return {
        "rms_mean": float(rms.mean()), "rms_p95": float(np.quantile(rms, 0.95)),
        "rms_max": float(rms.max()), "rms_std": float(rms.std()),
        "silence_fraction": float(np.mean(rms < 0.01)), "zcr_mean": float(zcr.mean()),
        "spectral_centroid_mean": float(centroid.mean()), "spectral_flux_mean": float(flux.mean()),
        "spectral_flux_p95": float(np.quantile(flux, 0.95)), "low_high_energy_ratio": low / (high + 1e-8),
    }


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    predicted = scores >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    return {"average_precision": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)), "precision_at_0_5": float(precision),
            "recall_at_0_5": float(recall), "f1_at_0_5": float(f1),
            "predicted_positive_at_0_5": int(predicted.sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia atributos acústicos simples em clipes TRAIN com LOMO.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["logistic_regression", "random_forest"], default=["logistic_regression", "random_forest"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    rows = read_csv(args.dataset)
    if not rows or any(row["split"] != "train" for row in rows):
        raise AssertionError("O dataset precisa ser exclusivamente TRAIN.")
    vectors, labels, matches = [], [], []
    prediction_rows = []
    for index, row in enumerate(rows, 1):
        start, end = float(row["clip_start_seconds"]), float(row["clip_end_seconds"])
        features = audio_features(decode_audio_clip(args.videos_dir / row["video_filename"], start, end))
        vectors.append([features[name] for name in FEATURES])
        labels.append(int(row["label"] == "GOAL")); matches.append(row["match_id"])
        prediction_rows.append({"clip_id": row["clip_id"], "match_id": row["match_id"], "label": row["label"],
                                "clip_start_seconds": row["clip_start_seconds"], "clip_end_seconds": row["clip_end_seconds"], **features})
        if index % 20 == 0 or index == len(rows):
            print(f"Clipes com áudio extraído: {index}/{len(rows)}", flush=True)
    x, y, groups = np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=int), np.asarray(matches)
    classifiers = {
        "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    predictions = []
    summary = {"experiment_id": args.output.name, "scope": "somente TRAIN; leave-one-match-out; TEST não acessado",
               "examples": len(rows), "labels": dict(Counter(row["label"] for row in rows)),
               "matches": len(set(matches)), "audio_sample_rate_hz": 8000, "audio_features": FEATURES,
               "clip_seconds": "herdados dos clipes etiquetados; sem reamostragem de vídeo",
               "models": {}, "warning": "Sinais acústicos auxiliares; não confirmam gol e não foram avaliados em VALIDATION ou TEST."}
    for model_name in args.models:
        scores = np.zeros(len(y), dtype=float)
        for held_match in sorted(set(groups)):
            test_mask, train_mask = groups == held_match, groups != held_match
            model = classifiers[model_name]()
            model.fit(x[train_mask], y[train_mask])
            scores[test_mask] = model.predict_proba(x[test_mask])[:, 1]
        summary["models"][model_name] = metrics(y, scores)
        for index, score in enumerate(scores):
            predictions.append({"model": model_name, "clip_id": rows[index]["clip_id"],
                                "match_id": matches[index], "label": rows[index]["label"], "score_goal": f"{score:.8f}"})
    args.output.mkdir(parents=True)
    write_csv(args.output / "audio_features_v1.csv", prediction_rows)
    write_csv(args.output / "predictions_lomo_v1.csv", predictions)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
