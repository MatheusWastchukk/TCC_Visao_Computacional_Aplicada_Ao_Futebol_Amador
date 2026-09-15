"""Aplica o verificador visual congelado a candidatos de um split de avaliação."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import cv2
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

from evaluate_goal_clip_pretrained_verifier_lomo_v1 import read_clip_frames


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def bounded_window(video_path: Path, start: float, end: float, durations: dict[Path, float]) -> tuple[float, float]:
    """Desloca a janela para caber no vídeo e preserva sua duração quando possível."""
    if video_path not in durations:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        finally:
            capture.release()
        if fps <= 0 or frames <= 0:
            raise ValueError(f"Metadados inválidos para {video_path}.")
        durations[video_path] = frames / fps
    duration, width = durations[video_path], end - start
    if duration <= width:
        return 0.0, duration
    bounded_start = min(max(0.0, start), duration - width)
    return bounded_start, bounded_start + width


def main() -> int:
    parser = argparse.ArgumentParser(description="Pontua candidatos com o verificador visual final, sem recalibração.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["train", "validation"], required=True)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")

    candidates, fields = read_csv(args.candidates)
    required = {"event_id", "match_id", "video_filename", "split", "candidate_start_seconds", "candidate_end_seconds", "rank_score"}
    if missing := required - set(fields):
        raise ValueError(f"Candidatos sem colunas: {sorted(missing)}")
    if not candidates or any(row["split"] != args.expected_split for row in candidates):
        raise AssertionError("O verificador exige candidatos de um único split esperado.")
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"match_id", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Catálogo sem colunas: {sorted(missing)}")
    if any(row["split"] != args.expected_split for row in catalog):
        raise AssertionError("O catálogo contém split diferente do esperado.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    frozen = joblib.load(args.model)
    if frozen.get("frames_per_clip") != 8 or frozen.get("embedding_dimensions") != 512:
        raise ValueError("Contrato incompatível do verificador visual congelado.")
    torch.set_num_threads(args.torch_threads)
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval()

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    bounded_candidates: list[dict[str, str]] = []
    durations: dict[Path, float] = {}
    pending_batches: list[torch.Tensor] = []
    for index, source in enumerate(candidates, start=1):
        row = dict(source)
        start, end = bounded_window(
            args.videos_dir / row["video_filename"], float(row["candidate_start_seconds"]),
            float(row["candidate_end_seconds"]), durations,
        )
        row["candidate_start_seconds"], row["candidate_end_seconds"] = f"{start:.6f}", f"{end:.6f}"
        frames = read_clip_frames(args.videos_dir / row["video_filename"], start, end, 8)
        pending_batches.append(torch.stack([transform(frame) for frame in frames]))
        labels.append(int(any(start <= goal <= end for goal in goals_by_match[row["match_id"]])))
        bounded_candidates.append(row)
        if len(pending_batches) == 8 or index == len(candidates):
            with torch.inference_mode():
                encoded = encoder(torch.cat(pending_batches)).cpu().numpy().astype(np.float32)
            embeddings.extend(encoded[offset * 8:(offset + 1) * 8].reshape(-1) for offset in range(len(pending_batches)))
            pending_batches.clear()
        if index % 8 == 0 or index == len(candidates):
            print(f"Clipes pontuados: {index}/{len(candidates)}", flush=True)

    scores = frozen["model"].predict_proba(np.stack(embeddings))[:, 1]
    rows: list[dict[str, object]] = []
    for candidate, label, score in zip(bounded_candidates, labels, scores):
        rows.append({**candidate, "catalog_goal_in_clip": str(bool(label)), "visual_score_goal": f"{float(score):.8f}"})
    args.output.mkdir(parents=True)
    write_csv(args.output / "candidate_visual_scores_v1.csv", rows, [*fields, "catalog_goal_in_clip", "visual_score_goal"])
    labels_array, scores_array = np.array(labels, dtype=int), np.array(scores, dtype=float)
    summary = {
        "experiment_id": args.output.name, "scope": "VALIDATION; configuração TRAIN congelada",
        "candidates": len(rows), "candidates_with_catalog_goal": int(labels_array.sum()),
        "encoder": "ResNet18 ImageNet congelada", "classifier": "regressão logística L2 congelada",
        "average_precision": float(average_precision_score(labels_array, scores_array)) if labels_array.any() else None,
        "roc_auc": float(roc_auc_score(labels_array, scores_array)) if len(np.unique(labels_array)) == 2 else None,
        "warning": "Escores descritivos sem ajuste de limiar em VALIDATION; não constituem filtro operacional binário.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
