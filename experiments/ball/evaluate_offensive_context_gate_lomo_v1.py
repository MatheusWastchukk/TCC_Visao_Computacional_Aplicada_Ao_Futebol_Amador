"""Baseline visual-temporal do Context Gate, avaliado por partida em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_window_frames(video_path: Path, start: float, end: float, count: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir vídeo TRAIN: {video_path}")
    try:
        frames: list[Image.Image] = []
        for timestamp in np.linspace(start, end, count, endpoint=False):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Não foi possível ler {video_path} em {timestamp:.3f}s.")
            frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        capture.release()


def visual_temporal_embedding(
    video_path: Path, start: float, end: float, frame_count: int, transform: object, encoder: torch.nn.Module,
) -> np.ndarray:
    frames = read_window_frames(video_path, start, end, frame_count)
    batch = torch.stack([transform(frame) for frame in frames])
    with torch.inference_mode():
        features = encoder(batch).cpu().numpy()
    return np.concatenate((
        features.mean(axis=0), features.std(axis=0), features[-1] - features[0],
        np.abs(np.diff(features, axis=0)).mean(axis=0),
    )).astype(np.float32)


def merged_seconds(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def metrics(rows: list[dict[str, str]], keep: np.ndarray) -> dict[str, float | int | None]:
    labels = np.asarray([row["context_decision"] == "KEEP_FOR_REVIEW" for row in rows])
    reject = ~labels
    decisions = keep.astype(bool)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, decisions, average="binary", zero_division=0)
    score_placeholder = np.asarray([float(row["score_keep"]) for row in rows])
    goal_linked = np.asarray([row["goal_linked"] == "True" for row in rows])
    intervals_by_match: dict[str, list[tuple[float, float]]] = defaultdict(list)
    kept_by_match: Counter[str] = Counter()
    for row, selected in zip(rows, decisions):
        if selected:
            intervals_by_match[row["match_id"]].append((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])))
            kept_by_match[row["match_id"]] += 1
    all_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        all_intervals[row["match_id"]].append((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])))
    before = sum(merged_seconds(value) for value in all_intervals.values())
    after = sum(merged_seconds(value) for value in intervals_by_match.values())
    return {
        "examples": int(len(rows)), "keep_examples": int(labels.sum()), "reject_examples": int(reject.sum()),
        "average_precision": float(average_precision_score(labels, score_placeholder)),
        "roc_auc": float(roc_auc_score(labels, score_placeholder)),
        "precision_keep": float(precision), "recall_keep": float(recall), "f1_keep": float(f1),
        "negative_rejection_rate": float((~decisions[reject]).mean()) if reject.any() else None,
        "goal_linked_keep_recall": float(decisions[goal_linked].mean()) if goal_linked.any() else None,
        "goal_linked_examples": int(goal_linked.sum()),
        "unique_review_seconds_before": float(before), "unique_review_seconds_after": float(after),
        "review_time_reduction": float(1.0 - after / before) if before else None,
        "segments_per_match_before": float(len(rows) / len(all_intervals)),
        "segments_per_match_after": float(decisions.sum() / len(all_intervals)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia Context Gate visual-temporal por leave-one-match-out em TRAIN.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-window", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.frames_per_window < 2:
        raise ValueError("São necessários ao menos dois frames por janela.")
    rows = read_csv(args.dataset)
    required = {"context_id", "match_id", "video_filename", "delivery_start_seconds", "delivery_end_seconds", "rank_score", "goal_linked", "context_decision", "annotation_status"}
    if not rows or any(field not in rows[0] for field in required):
        raise ValueError("Dataset de contexto incompleto.")
    if any(row["annotation_status"] != "completed" for row in rows):
        raise ValueError("O baseline exige todas as anotações concluídas.")
    if {row["context_decision"] for row in rows} != {"KEEP_FOR_REVIEW", "REJECT_NO_ACTION"}:
        raise ValueError("As decisões devem ser KEEP_FOR_REVIEW ou REJECT_NO_ACTION.")
    torch.set_num_threads(args.torch_threads)
    transform = ResNet18_Weights.DEFAULT.transforms()
    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    embeddings: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        embeddings.append(visual_temporal_embedding(
            args.videos_dir / row["video_filename"], float(row["delivery_start_seconds"]),
            float(row["delivery_end_seconds"]), args.frames_per_window, transform, encoder,
        ))
        if index % 25 == 0 or index == len(rows):
            print(f"Janelas processadas: {index}/{len(rows)}", flush=True)
    visual_x = np.stack(embeddings)
    rank_x = np.asarray([[float(row["rank_score"]), float(row["temporal_rank"])] for row in rows], dtype=np.float32)
    x = np.column_stack((visual_x, rank_x))
    y = np.asarray([row["context_decision"] == "KEEP_FOR_REVIEW" for row in rows], dtype=int)
    groups = np.asarray([row["match_id"] for row in rows])
    scores = np.zeros(len(rows), dtype=np.float64)
    for held_match in sorted(set(groups)):
        train_mask, test_mask = groups != held_match, groups == held_match
        model = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, solver="liblinear",
        ))
        model.fit(x[train_mask], y[train_mask])
        scores[test_mask] = model.predict_proba(x[test_mask])[:, 1]
    predicted = [{**row, "score_keep": f"{score:.8f}", "decision_at_0_5": "KEEP_FOR_REVIEW" if score >= 0.5 else "REJECT_NO_ACTION"} for row, score in zip(rows, scores)]
    output_rows = [{**row, "score_keep": item["score_keep"]} for row, item in zip(rows, predicted)]
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN; leave-one-match-out por partida",
        "target": "KEEP_FOR_REVIEW versus REJECT_NO_ACTION; não é decisão oficial de gol",
        "encoder": "ResNet18 ImageNet congelada", "frames_per_window": args.frames_per_window,
        "features": int(x.shape[1]), "matches": int(len(set(groups))),
        "labels": dict(Counter(row["context_label"] for row in rows)),
        "metrics_at_threshold_0_5": metrics(output_rows, scores >= 0.5),
        "warning": "O limiar 0,5 é apenas baseline de desenvolvimento. A elegibilidade de confirmação depende de preservar todos os candidatos ligados a gol catalogado e de redução mensurável do tempo de revisão.",
    }
    args.output.mkdir(parents=True)
    with (args.output / "context_gate_predictions_lomo_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predicted[0])); writer.writeheader(); writer.writerows(predicted)
    with (args.output / "context_embeddings_lomo_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["context_id", "match_id", "context_decision"] + [f"feature_{index:04d}" for index in range(x.shape[1])])
        for row, vector in zip(rows, x):
            writer.writerow([row["context_id"], row["match_id"], row["context_decision"]] + [f"{value:.8f}" for value in vector])
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics_at_threshold_0_5"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
