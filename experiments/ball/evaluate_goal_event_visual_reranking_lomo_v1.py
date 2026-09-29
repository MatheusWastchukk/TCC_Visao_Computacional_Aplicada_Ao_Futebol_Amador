"""Avalia reranking visual e fusão simples por partida, somente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

from evaluate_goal_clip_pretrained_verifier_lomo_v1 import read_clip_frames


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_diverse(rows: list[dict[str, str]], limit: int, separation: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def covers(row: dict[str, object], goal_time: float) -> bool:
    return float(row["candidate_start_seconds"]) <= goal_time <= float(row["candidate_end_seconds"])


def bounded_window(video_path: Path, start: float, end: float, durations: dict[Path, float]) -> tuple[float, float]:
    """Mantém a duração do clipe ao deslocar janelas próximas às bordas do vídeo."""
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
    duration = durations[video_path]
    width = end - start
    if duration <= width:
        return 0.0, duration
    bounded_start = min(max(0.0, start), duration - width)
    return bounded_start, bounded_start + width


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia reranking visual LOMO sobre candidatos TRAIN do ranker.")
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--verifier-dataset", type=Path, required=True)
    parser.add_argument("--sequence-embeddings", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preselect-k", type=int, default=20)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    parser.add_argument("--delivery-pre-seconds", type=float, default=20.0)
    parser.add_argument("--delivery-post-seconds", type=float, default=40.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--embedding-checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    limits = sorted(set(args.candidate_limits))
    if args.preselect_k < max(limits):
        raise ValueError("preselect-k deve ser ao menos o maior candidate-limit.")

    prediction_rows, prediction_fields = read_csv(args.rank_predictions)
    required_predictions = {"model", "group_uid", "match_id", "best_timestamp_seconds", "candidate_start_seconds", "candidate_end_seconds", "rank_score"}
    if missing := required_predictions - set(prediction_fields):
        raise ValueError(f"Predições do ranker sem colunas: {sorted(missing)}")
    rank_rows = [row for row in prediction_rows if row["model"] == "random_forest"]
    if not rank_rows:
        raise ValueError("Não há predições random_forest para reranking.")

    split_rows, split_fields = read_csv(args.split_manifest)
    if {"match_id", "video_filename", "split"} - set(split_fields):
        raise ValueError("Manifesto de split incompleto.")
    videos = {row["match_id"]: row["video_filename"] for row in split_rows if row["split"] == "train"}
    if {row["match_id"] for row in rank_rows} - set(videos):
        raise AssertionError("Há predições fora de TRAIN.")

    catalog, catalog_fields = read_csv(args.goal_timestamps)
    required_catalog = {"goal_catalog_id", "match_id", "split", "annotation_status", "event_timestamp_seconds"}
    if missing := required_catalog - set(catalog_fields):
        raise ValueError(f"Catálogo sem colunas: {sorted(missing)}")
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("O reranking aceita somente timestamps TRAIN; VALIDATION e TEST permanecem fechados.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    verifier_rows, verifier_fields = read_csv(args.verifier_dataset)
    required_verifier = {"clip_id", "match_id", "split", "label"}
    if missing := required_verifier - set(verifier_fields):
        raise ValueError(f"Dataset do verificador sem colunas: {sorted(missing)}")
    if any(row["split"] != "train" for row in verifier_rows):
        raise AssertionError("O dataset do verificador deve conter somente TRAIN.")
    archive = np.load(args.sequence_embeddings, allow_pickle=False)
    embeddings_by_id = {clip_id: embedding.reshape(-1) for clip_id, embedding in zip(archive["clip_ids"], archive["embeddings"])}
    if {row["clip_id"] for row in verifier_rows} - set(embeddings_by_id):
        raise ValueError("Embeddings ausentes para o dataset do verificador.")
    verifier_x = np.stack([embeddings_by_id[row["clip_id"]] for row in verifier_rows])
    verifier_y = np.asarray([int(row["label"] == "GOAL") for row in verifier_rows], dtype=int)
    verifier_matches = np.asarray([row["match_id"] for row in verifier_rows])

    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rank_rows:
        by_match[row["match_id"]].append(row)
    candidates: list[dict[str, object]] = []
    for match_id in sorted(by_match):
        for index, row in enumerate(select_diverse(by_match[match_id], args.preselect_k, args.min_separation_seconds), start=1):
            start, end = float(row["candidate_start_seconds"]), float(row["candidate_end_seconds"])
            candidates.append({
                "candidate_id": f"{match_id}__rf_lomo_top{index:02d}", "match_id": match_id,
                "video_filename": videos[match_id], "group_uid": row["group_uid"],
                "candidate_start_seconds": f"{start:.6f}", "candidate_end_seconds": f"{end:.6f}",
                "delivery_start_seconds": f"{max(0.0, float(row['best_timestamp_seconds']) - args.delivery_pre_seconds):.6f}",
                "delivery_end_seconds": f"{float(row['best_timestamp_seconds']) + args.delivery_post_seconds:.6f}",
                "rank_score": f"{float(row['rank_score']):.8f}",
            })
    if not candidates:
        raise ValueError("Nenhum candidato foi selecionado.")

    durations: dict[Path, float] = {}
    for candidate in candidates:
        start, end = bounded_window(
            args.videos_dir / str(candidate["video_filename"]),
            float(candidate["candidate_start_seconds"]),
            float(candidate["candidate_end_seconds"]), durations,
        )
        candidate["candidate_start_seconds"] = f"{start:.6f}"
        candidate["candidate_end_seconds"] = f"{end:.6f}"
        candidate["catalog_goal_in_clip"] = int(any(start <= goal <= end for goal in goals_by_match[str(candidate["match_id"])]))

    torch.set_num_threads(args.torch_threads)
    transform = ResNet18_Weights.DEFAULT.transforms()
    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    checkpoint_path = args.embedding_checkpoint or args.output.with_name(args.output.name + "_embeddings_checkpoint.npy")
    progress_path = checkpoint_path.with_suffix(".json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    expected_candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if checkpoint_path.exists() or progress_path.exists():
        if not checkpoint_path.exists() or not progress_path.exists():
            raise FileExistsError("Checkpoint incompleto: arquivo de embeddings e manifesto devem existir juntos.")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("candidate_ids") != expected_candidate_ids:
            raise ValueError("A ordem dos candidatos difere do checkpoint existente.")
        candidate_embeddings = np.load(checkpoint_path, mmap_mode="r+")
        if candidate_embeddings.shape != (len(candidates), 512 * 8):
            raise ValueError("Dimensoes do checkpoint de embeddings inesperadas.")
        completed = int(progress["completed_candidates"])
        if not 0 <= completed <= len(candidates):
            raise ValueError("Progresso invalido no checkpoint de embeddings.")
        print(f"Retomando embeddings: {completed}/{len(candidates)} ja concluidos.", flush=True)
    else:
        candidate_embeddings = np.lib.format.open_memmap(
            checkpoint_path, mode="w+", dtype=np.float32, shape=(len(candidates), 512 * 8)
        )
        completed = 0
        progress = {"candidate_ids": expected_candidate_ids, "completed_candidates": 0}
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
    for index, candidate in enumerate(candidates, start=1):
        if index <= completed:
            continue
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                frames = read_clip_frames(args.videos_dir / str(candidate["video_filename"]), float(candidate["candidate_start_seconds"]), float(candidate["candidate_end_seconds"]), 8)
                batch = torch.stack([transform(frame) for frame in frames])
                with torch.inference_mode():
                    candidate_embeddings[index - 1] = encoder(batch).cpu().numpy().astype(np.float32).reshape(-1)
                last_error = None
                break
            except (RuntimeError, FileNotFoundError, cv2.error) as error:
                last_error = error
                if attempt < 2:
                    print(f"Falha temporaria no candidato {index}; nova tentativa {attempt + 2}/3: {error}", flush=True)
                    time.sleep(2)
        if last_error is not None:
            candidate_embeddings.flush()
            raise last_error
        if index % 25 == 0 or index == len(candidates):
            candidate_embeddings.flush()
            progress = {"candidate_ids": expected_candidate_ids, "completed_candidates": index}
            progress_tmp = progress_path.with_suffix(".tmp")
            progress_tmp.write_text(json.dumps(progress), encoding="utf-8")
            progress_tmp.replace(progress_path)
            print(f"Embeddings de candidatos: {index}/{len(candidates)}", flush=True)
    candidate_x = candidate_embeddings
    candidate_matches = np.asarray([str(row["match_id"]) for row in candidates])
    visual_scores = np.zeros(len(candidates), dtype=np.float64)
    for held_match in sorted(set(candidate_matches)):
        train_mask, test_mask = verifier_matches != held_match, candidate_matches == held_match
        verifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
        verifier.fit(verifier_x[train_mask], verifier_y[train_mask])
        visual_scores[test_mask] = verifier.predict_proba(candidate_x[test_mask])[:, 1]
    for candidate, score in zip(candidates, visual_scores):
        candidate["visual_score_goal"] = f"{float(score):.8f}"

    candidate_y = np.asarray([int(row["catalog_goal_in_clip"]) for row in candidates], dtype=int)
    rank_scores = np.asarray([float(row["rank_score"]) for row in candidates], dtype=np.float64)
    fusion_x = np.column_stack([rank_scores, visual_scores])
    fusion_scores = np.zeros(len(candidates), dtype=np.float64)
    for held_match in sorted(set(candidate_matches)):
        train_mask, test_mask = candidate_matches != held_match, candidate_matches == held_match
        fusion = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))
        fusion.fit(fusion_x[train_mask], candidate_y[train_mask])
        fusion_scores[test_mask] = fusion.predict_proba(fusion_x[test_mask])[:, 1]
    for candidate, score in zip(candidates, fusion_scores):
        candidate["fusion_score_goal"] = f"{float(score):.8f}"

    comparison_rows: list[dict[str, object]] = []
    for score_name in ("rank_score", "visual_score_goal", "fusion_score_goal"):
        for limit in limits:
            covered = 0
            for match_id, goals in goals_by_match.items():
                selected = sorted([row for row in candidates if row["match_id"] == match_id], key=lambda item: float(item[score_name]), reverse=True)[:limit]
                covered += sum(int(any(covers(row, goal) for row in selected)) for goal in goals)
            comparison_rows.append({"method": score_name, "k": limit, "catalog_goals": sum(len(goals) for goals in goals_by_match.values()), "covered_goals": covered, "recall_at_k": f"{covered / sum(len(goals) for goals in goals_by_match.values()):.6f}"})

    args.output.mkdir(parents=True)
    candidate_fields = list(candidates[0])
    write_csv(args.output / "candidate_scores_lomo_v1.csv", candidates, candidate_fields)
    write_csv(args.output / "recall_at_k_comparison_v1.csv", comparison_rows, ["method", "k", "catalog_goals", "covered_goals", "recall_at_k"])
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN; todos os escores por partida out-of-fold",
        "ranker": "Random Forest temporal LOMO", "preselect_k": args.preselect_k,
        "visual_verifier": "ResNet18 congelada + regressão logística LOMO", "fusion": "regressão logística L2 com rank_score e visual_score_goal, LOMO",
        "candidates": len(candidates), "candidates_with_goal": int(candidate_y.sum()),
        "warning": "Experimento de desenvolvimento; não seleciona limiar e não usa VALIDATION ou TEST.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
