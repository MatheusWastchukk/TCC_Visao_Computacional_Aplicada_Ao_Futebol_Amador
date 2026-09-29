"""Ranqueia clips TRAIN com classificacao LOMO de subclips de 5 s e ResNet18 congelada."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frame_embeddings(video_path: Path, timestamps: list[float], transform: object,
                     encoder: torch.nn.Module) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    frames: list[Image.Image] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Nao foi possivel ler {video_path} em {timestamp:.3f}s")
            frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    batch = torch.stack([transform(frame) for frame in frames])
    with torch.inference_mode():
        return encoder(batch).cpu().numpy().astype(np.float32)


def interval_times(start: float, end: float, count: int) -> list[float]:
    if end <= start:
        raise ValueError(f"Intervalo invalido: {start}..{end}")
    width = (end - start) / count
    return [start + (index + 0.5) * width for index in range(count)]


def metric_row(method: str, ranked: dict[str, list[dict[str, object]]],
               goals: dict[str, list[float]], limit: int) -> dict[str, object]:
    selected = {match: rows[:limit] for match, rows in ranked.items()}
    covered = sum(
        any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in selected[match])
        for match, timestamps in goals.items() for goal in timestamps
    )
    segment_count = 0
    unique_seconds = 0.0
    for rows in selected.values():
        merged: list[list[float]] = []
        for start, end in sorted((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in rows):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segment_count += len(merged)
        unique_seconds += sum(end - start for start, end in merged)
    total_goals = sum(map(len, goals.values()))
    return {"method": method, "k": limit, "catalog_goals": total_goals,
            "covered_goals": covered, "recall_at_k": f"{covered / total_goals:.6f}",
            "candidates": sum(map(len, selected.values())), "merged_segments": segment_count,
            "unique_review_minutes_per_match": f"{unique_seconds / 60 / len(selected):.3f}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Subclips de 5 s no pool TRAIN completo; LOMO por partida.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--verifier-dataset", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    candidates, candidate_fields = read_csv(args.candidate_scores)
    verifier_rows, verifier_fields = read_csv(args.verifier_dataset)
    catalog, catalog_fields = read_csv(args.goal_timestamps)
    split_rows, split_fields = read_csv(args.split_manifest)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(candidates) != 5742 or any(row["split"] != "train" for row in catalog):
        raise AssertionError("Esperados 5.742 candidatos e catalogo exclusivamente TRAIN.")
    if any(row["split"] != "train" for row in verifier_rows):
        raise AssertionError("Dataset do verificador precisa ser exclusivamente TRAIN.")
    if {"match_id", "video_filename", "split"} - set(split_fields):
        raise ValueError("Manifesto de split sem colunas obrigatorias.")
    video_by_match = {row["match_id"]: row["video_filename"] for row in split_rows if row["split"] == "train"}
    goals: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    durations: dict[str, float] = {}
    for match_id, filename in video_by_match.items():
        capture = cv2.VideoCapture(str(args.videos_dir / filename))
        if not capture.isOpened():
            raise FileNotFoundError(args.videos_dir / filename)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        capture.release()
        if fps <= 0 or frames <= 0:
            raise ValueError(f"Metadados invalidos: {filename}")
        durations[match_id] = frames / fps

    # Create exactly six 5-second supervision bins per 30-second verifier clip.
    train_segments: list[dict[str, object]] = []
    train_matches: list[str] = []
    train_labels: list[int] = []
    train_features: list[np.ndarray] = []
    torch.set_num_threads(args.torch_threads)
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    for clip_index, row in enumerate(verifier_rows, start=1):
        start, end = float(row["clip_start_seconds"]), float(row["clip_end_seconds"])
        edges = np.linspace(start, end, 7)
        timestamps = interval_times(start, end, 6)
        for subclip in range(6):
            left, right = float(edges[subclip]), float(edges[subclip + 1])
            positive = int(any(left <= goal < right or (subclip == 5 and goal == right)
                               for goal in goals[row["match_id"]]))
            train_segments.append({"clip_id": row["clip_id"], "match_id": row["match_id"],
                                   "subclip_index": subclip, "start_seconds": left,
                                   "end_seconds": right, "goal_event_target": positive})
            train_matches.append(row["match_id"])
            train_labels.append(positive)
        for attempt in range(3):
            try:
                encoded = frame_embeddings(args.videos_dir / row["video_filename"], timestamps, transform, encoder)
                train_features.extend(encoded)
                break
            except (RuntimeError, FileNotFoundError, cv2.error):
                if attempt == 2:
                    raise
                time.sleep(1)
        if clip_index % 50 == 0 or clip_index == len(verifier_rows):
            print(f"Subclips de treino: {clip_index}/{len(verifier_rows)} clipes", flush=True)
    train_x = np.asarray(train_features, dtype=np.float32)
    train_y = np.asarray(train_labels, dtype=np.int32)
    train_match_array = np.asarray(train_matches)
    if len(np.unique(train_y)) != 2:
        raise ValueError("Supervisao de subclips precisa conter positivos e negativos.")

    candidate_ids = [row["candidate_id"] for row in candidates]
    checkpoint_path = args.output.with_name(args.output.name + "_subclip_embeddings.npy")
    progress_path = checkpoint_path.with_suffix(".json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(candidates), 12, 512)
    if checkpoint_path.exists() or progress_path.exists():
        if not checkpoint_path.exists() or not progress_path.exists():
            raise FileExistsError("Checkpoint de subclips incompleto.")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("candidate_ids") != candidate_ids:
            raise ValueError("Ordem de candidatos nao corresponde ao checkpoint.")
        candidate_x = np.load(checkpoint_path, mmap_mode="r+")
        if candidate_x.shape != shape:
            raise ValueError(f"Shape inesperado no checkpoint: {candidate_x.shape}")
        completed = int(progress["completed_candidates"])
        print(f"Retomando features multi-subclip: {completed}/{len(candidates)}", flush=True)
    else:
        candidate_x = np.lib.format.open_memmap(checkpoint_path, mode="w+", dtype=np.float32, shape=shape)
        completed = 0
        progress = {"candidate_ids": candidate_ids, "completed_candidates": 0}
        progress_path.write_text(json.dumps(progress), encoding="utf-8")

    candidate_matches = np.asarray([row["match_id"] for row in candidates])
    for index, row in enumerate(candidates, start=1):
        if index <= completed:
            continue
        center = float(row["delivery_end_seconds"]) - 40.0
        duration = durations[row["match_id"]]
        start, end = max(0.0, center - 20.0), min(duration, center + 40.0)
        timestamps = interval_times(start, end, 12)
        for attempt in range(3):
            try:
                candidate_x[index - 1] = frame_embeddings(args.videos_dir / row["video_filename"], timestamps, transform, encoder)
                break
            except (RuntimeError, FileNotFoundError, cv2.error) as error:
                if attempt == 2:
                    candidate_x.flush()
                    raise
                print(f"Falha temporaria no candidato {index}; nova tentativa {attempt + 2}/3: {error}", flush=True)
                time.sleep(2)
        if index % args.checkpoint_every == 0 or index == len(candidates):
            candidate_x.flush()
            progress = {"candidate_ids": candidate_ids, "completed_candidates": index}
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress), encoding="utf-8")
            temporary.replace(progress_path)
            print(f"Features multi-subclip: {index}/{len(candidates)}", flush=True)

    candidate_rows: list[dict[str, object]] = []
    subclip_rows: list[dict[str, object]] = []
    candidate_targets: list[int] = []
    for index, row in enumerate(candidates):
        center = float(row["delivery_end_seconds"]) - 40.0
        start, end = max(0.0, center - 20.0), min(durations[row["match_id"]], center + 40.0)
        scores_target = int(any(start <= goal <= end for goal in goals[row["match_id"]]))
        candidate_targets.append(scores_target)
        candidate_rows.append({**row, "delivery_start_seconds": f"{start:.6f}",
                               "delivery_end_seconds": f"{end:.6f}", "catalog_goal_in_delivery": scores_target})

    raw_subclip_scores = np.zeros(shape[:2], dtype=np.float32)
    for held_match in sorted(set(candidate_matches)):
        train_mask = train_match_array != held_match
        candidate_mask = candidate_matches == held_match
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
        model.fit(train_x[train_mask], train_y[train_mask])
        raw_subclip_scores[candidate_mask] = model.predict_proba(np.asarray(candidate_x[candidate_mask]).reshape(-1, 512))[:, 1].reshape(-1, 12)

    aggregate_scores: dict[str, np.ndarray] = {
        "subclip_max_score": raw_subclip_scores.max(axis=1),
        "subclip_top2_mean_score": np.sort(raw_subclip_scores, axis=1)[:, -2:].mean(axis=1),
        "subclip_top3_mean_score": np.sort(raw_subclip_scores, axis=1)[:, -3:].mean(axis=1),
        "subclip_active_bins_ge_0_5": (raw_subclip_scores >= 0.5).sum(axis=1).astype(np.float32),
    }
    original_rank = np.asarray([float(row["rank_score"]) for row in candidates])
    target_array = np.asarray(candidate_targets, dtype=np.int32)
    candidate_fusion_scores: dict[str, np.ndarray] = {}
    for feature_name in ("subclip_max_score", "subclip_top2_mean_score", "subclip_top3_mean_score"):
        fused = np.zeros(len(candidates), dtype=np.float64)
        matrix = np.column_stack([original_rank, aggregate_scores[feature_name]])
        for held_match in sorted(set(candidate_matches)):
            train_mask, test_mask = candidate_matches != held_match, candidate_matches == held_match
            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
            model.fit(matrix[train_mask], target_array[train_mask])
            fused[test_mask] = model.predict_proba(matrix[test_mask])[:, 1]
        candidate_fusion_scores[f"fusion_{feature_name}"] = fused

    for index, row in enumerate(candidate_rows):
        for name, values in aggregate_scores.items():
            row[name] = f"{float(values[index]):.8f}"
        for name, values in candidate_fusion_scores.items():
            row[name] = f"{float(values[index]):.8f}"
        for subclip, score in enumerate(raw_subclip_scores[index]):
            left, right = interval_times(float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"]), 12)[subclip], 0.0
            width = (float(row["delivery_end_seconds"]) - float(row["delivery_start_seconds"])) / 12
            right = left + width
            subclip_rows.append({"candidate_id": row["candidate_id"], "match_id": row["match_id"],
                                 "subclip_index": subclip, "start_seconds": f"{left:.6f}",
                                 "end_seconds": f"{right:.6f}", "lomo_goal_event_score": f"{float(score):.8f}"})

    ranked_rows: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    all_score_fields = [*aggregate_scores, *candidate_fusion_scores]
    for match in sorted(set(candidate_matches)):
        match_rows = [row for row in candidate_rows if row["match_id"] == match]
        ranked_rows["temporal_rank_score"][match] = sorted(match_rows, key=lambda row: float(row["rank_score"]), reverse=True)
        for field in all_score_fields:
            ranked_rows[field][match] = sorted(match_rows, key=lambda row: float(row[field]), reverse=True)
        for feature_name in ("subclip_max_score", "subclip_top2_mean_score"):
            ranks: dict[str, float] = defaultdict(float)
            for field in ("rank_score", feature_name):
                for rank, item in enumerate(sorted(match_rows, key=lambda row: float(row[field]), reverse=True), 1):
                    ranks[str(item["group_uid"])] += 1.0 / (60 + rank)
            ranked_rows[f"rrf_temporal_{feature_name}"][match] = sorted(match_rows, key=lambda item: ranks[str(item["group_uid"])], reverse=True)

    metric_rows: list[dict[str, object]] = []
    for name, by_match in ranked_rows.items():
        diverse = {match: [] for match in by_match}
        for match, rows in by_match.items():
            chosen: list[dict[str, object]] = []
            for item in rows:
                center_item = float(item["delivery_end_seconds"]) - 40.0
                if all(abs(center_item - (float(other["delivery_end_seconds"]) - 40.0)) >= 45.0 for other in chosen):
                    chosen.append(item)
            diverse[match] = chosen
        for limit in (20, 30, 40, 50, 60, 80, 100):
            metric_rows.append(metric_row(name + "_gap45s", diverse, goals, limit))

    args.output.mkdir(parents=True)
    write_csv(args.output / "candidate_multisubclip_scores_lomo_v1.csv", candidate_rows)
    write_csv(args.output / "candidate_subclip_frame_scores_lomo_v1.csv", subclip_rows)
    write_csv(args.output / "training_subclip_targets_v1.csv", train_segments)
    write_csv(args.output / "recall_review_curve_lomo_v1.csv", metric_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 25-fold LOMO; 5,742 full-pool candidate windows; TEST not accessed",
        "encoder": "frozen ImageNet ResNet18, one frame sampled at the center of each 5-second subclip",
        "this_is_not": "a pretrained video backbone; temporal order is represented only by per-subclip scores and aggregation",
        "training_clips": len(verifier_rows), "training_subclips": len(train_segments),
        "positive_training_subclips": int(train_y.sum()), "negative_training_subclips": int(len(train_y) - train_y.sum()),
        "candidate_groups": len(candidates), "subclips_per_candidate": 12,
        "candidate_groups_with_delivery_goal": int(target_array.sum()),
        "goal_timestamps": sum(map(len, goals.values())),
        "window": "20 seconds before and 40 seconds after candidate center, truncated at video duration",
        "warning": "Exploratory subclip detector trained from exact catalog goal timestamps; no action-relevance labels and no TEST.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
