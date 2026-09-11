"""Prepara clipes TRAIN pareados para desenvolver o verificador visual de gols."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def clip_bounds(center: float, before: float, after: float) -> tuple[float, float]:
    return center - before, center + after


def bounded_clip_bounds(center: float, before: float, after: float, duration: float) -> tuple[float, float]:
    """Desloca a janela para caber no vídeo, sem alterar sua duração."""
    width = before + after
    if duration < width:
        raise ValueError("Vídeo curto demais para a janela solicitada.")
    start = min(max(0.0, center - before), duration - width)
    return start, start + width


def is_safe(center: float, goal_times: list[float], used_centers: list[float], exclusion: float, separation: float) -> bool:
    return (
        all(abs(center - goal_time) >= exclusion for goal_time in goal_times)
        and all(abs(center - used) >= separation for used in used_centers)
    )


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo TRAIN: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Metadados inválidos para o vídeo: {path}")
    return frame_count / fps


def choose_safe_center(
    goal_id: str,
    duration: float,
    goal_times: list[float],
    used_centers: list[float],
    before: float,
    after: float,
    exclusion: float,
    separation: float,
) -> float:
    minimum, maximum = before, duration - after
    if maximum <= minimum:
        raise ValueError("Vídeo curto demais para a janela solicitada.")
    candidates = [minimum + (maximum - minimum) * index / 1999 for index in range(2000)]
    offset = int(hashlib.sha256(goal_id.encode("utf-8")).hexdigest()[:8], 16) % len(candidates)
    for shift in range(len(candidates)):
        center = candidates[(offset + shift) % len(candidates)]
        if is_safe(center, goal_times, used_centers, exclusion, separation):
            return center
    raise ValueError(f"Não há janela negativa segura para {goal_id}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara dataset pareado TRAIN de clipes de gol e não gol.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-before", type=float, default=10.0)
    parser.add_argument("--seconds-after", type=float, default=20.0)
    parser.add_argument("--goal-exclusion-seconds", type=float, default=90.0)
    parser.add_argument("--negative-separation-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.seconds_before <= 0 or args.seconds_after <= 0:
        raise ValueError("As durações antes e depois devem ser positivas.")

    catalog = read_csv(args.catalog)
    if any(row["split"] != "train" for row in catalog):
        raise AssertionError("O catálogo do verificador aceita somente TRAIN; VALIDATION e TEST permanecem fechados.")
    positives = [row for row in catalog if row["annotation_status"] == "completed"]
    if not positives or any(not row["event_timestamp_seconds"] for row in positives):
        raise ValueError("O catálogo precisa conter gols completed com timestamp.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in positives:
        goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    candidates = read_csv(args.candidates)
    hard_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        if row["split"] != "train":
            raise AssertionError("Os candidatos do verificador devem ser somente TRAIN.")
        if row["annotation_status"] == "completed" and row["ground_truth"] == "NO_GOAL":
            hard_by_match[row["match_id"]].append(row)
    for rows in hard_by_match.values():
        rows.sort(key=lambda row: float(row.get("candidate_score") or 0.0), reverse=True)

    duration_by_video: dict[str, float] = {}
    used_centers: dict[str, list[float]] = defaultdict(list)
    hard_cursor: dict[str, int] = defaultdict(int)
    output_rows: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    for index, positive in enumerate(positives, start=1):
        match_id, filename = positive["match_id"], positive["video_filename"]
        if filename not in duration_by_video:
            duration_by_video[filename] = video_duration(args.videos_dir / filename)
        duration = duration_by_video[filename]
        goal_time = float(positive["event_timestamp_seconds"])
        start, end = bounded_clip_bounds(goal_time, args.seconds_before, args.seconds_after, duration)
        output_rows.append({
            "clip_id": f"goal_clip_v2_{index:04d}_positive", "pair_id": f"goal_clip_v2_pair_{index:04d}",
            "match_id": match_id, "video_filename": filename, "split": "train", "label": "GOAL",
            "clip_start_seconds": f"{start:.3f}", "clip_end_seconds": f"{end:.3f}",
            "center_seconds": f"{goal_time:.3f}", "source_type": "catalog_timestamp_positive",
            "source_id": positive["goal_catalog_id"], "review_status": "timestamp_confirmed",
            "notes": "Janela centrada no timestamp de gol revisado; uso exclusivo em desenvolvimento TRAIN.",
        })

        negative_center: float | None = None
        negative_source, negative_id = "safe_temporal_negative", ""
        for hard in hard_by_match[match_id][hard_cursor[match_id]:]:
            candidate_center = float(hard["event_timestamp_seconds"])
            if is_safe(candidate_center, goals_by_match[match_id], used_centers[match_id], args.goal_exclusion_seconds, args.negative_separation_seconds):
                negative_center, negative_source, negative_id = candidate_center, "reviewed_hard_negative", hard["candidate_uid"]
                hard_cursor[match_id] = hard_by_match[match_id].index(hard) + 1
                break
        if negative_center is None:
            negative_center = choose_safe_center(
                positive["goal_catalog_id"], duration, goals_by_match[match_id], used_centers[match_id],
                args.seconds_before, args.seconds_after, args.goal_exclusion_seconds, args.negative_separation_seconds,
            )
            negative_id = f"safe_negative_for_{positive['goal_catalog_id']}"
        used_centers[match_id].append(negative_center)
        start, end = bounded_clip_bounds(negative_center, args.seconds_before, args.seconds_after, duration)
        output_rows.append({
            "clip_id": f"goal_clip_v2_{index:04d}_negative", "pair_id": f"goal_clip_v2_pair_{index:04d}",
            "match_id": match_id, "video_filename": filename, "split": "train", "label": "NO_GOAL",
            "clip_start_seconds": f"{start:.3f}", "clip_end_seconds": f"{end:.3f}",
            "center_seconds": f"{negative_center:.3f}", "source_type": negative_source,
            "source_id": negative_id,
            "review_status": "human_confirmed" if negative_source == "reviewed_hard_negative" else "pending_sample_audit",
            "notes": "Negativo pareado na mesma partida, fora da zona de exclusão de todos os gols catalogados.",
        })
        source_counts[negative_source] += 1
    args.output.mkdir(parents=True)
    fields = list(output_rows[0])
    with (args.output / "goal_clip_verifier_dataset_v2.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN", "pairs": len(positives),
        "clips": len(output_rows), "labels": dict(Counter(row["label"] for row in output_rows)),
        "negative_sources": dict(source_counts), "seconds_before": args.seconds_before,
        "seconds_after": args.seconds_after, "goal_exclusion_seconds": args.goal_exclusion_seconds,
        "negative_separation_seconds": args.negative_separation_seconds,
        "excluded_catalog_rows": dict(Counter(row["annotation_status"] for row in catalog if row["annotation_status"] != "completed")),
        "warning": "Negativos temporais seguros exigem auditoria humana amostral antes de treinamento ou métricas de classificação.",
    }
    (args.output / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
