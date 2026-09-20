"""Gera candidatos de evento em vídeos TRAIN por coocorrência bola–estrutura do gol.

Este script é um gerador de hipóteses de alta cobertura. Ele não decide se houve
gol: a decisão humana posterior deve usar GOAL, NO_GOAL ou NOT_EVALUABLE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


EVENT_FIELDS = [
    "event_id", "source_file", "source_interval_id", "source_interval_label",
    "match_id", "video_filename", "split", "event_timestamp_seconds",
    "event_time_hhmmss", "ground_truth", "ball_visible", "goal_visible",
    "spatial_candidate", "relative_motion_valid", "identity_stable",
    "center_visible_after", "ball_at_center_after", "players_reset",
    "kickoff_observed", "restart_time", "scoreboard_visible", "score_changed",
    "not_evaluable_reason", "annotation_status", "notes", "candidate_score",
    "candidate_hit_frames", "candidate_start_seconds", "candidate_end_seconds",
]
FRAME_FIELDS = [
    "match_id", "video_filename", "timestamp_seconds", "frame_index",
    "goal_box_json", "goal_confidence", "ball_box_json", "ball_confidence",
    "ball_inside_expanded_goal", "normalized_distance", "candidate_score",
]


@dataclass(frozen=True)
class Hit:
    match_id: str
    video_filename: str
    timestamp: float
    frame_index: int
    goal_box: tuple[float, float, float, float]
    goal_confidence: float
    ball_box: tuple[float, float, float, float]
    ball_confidence: float
    inside_expanded_goal: bool
    normalized_distance: float
    score: float


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def expand_box(
    box: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return x1 - width * margin, y1 - height * margin, x2 + width * margin, y2 + height * margin


def inside(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def normalized_distance(
    first: tuple[float, float], second: tuple[float, float, float, float]
) -> float:
    width, height = second[2] - second[0], second[3] - second[1]
    diagonal = max(1.0, math.hypot(width, height))
    goal_center = box_center(second)
    return math.hypot(first[0] - goal_center[0], first[1] - goal_center[1]) / diagonal


def load_videos(split_manifest: Path, videos_dir: Path, include_matches: list[str] | None, expected_split: str) -> list[dict[str, str]]:
    with split_manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    selected = set(include_matches) if include_matches else None
    if selected:
        known = {row["match_id"] for row in rows}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Partidas ausentes do manifesto de split: {unknown}")
    selected_rows = []
    for row in rows:
        if selected and row["match_id"] not in selected:
            continue
        if row["split"] != expected_split:
            if selected and row["match_id"] in selected:
                raise AssertionError(
                    f"{row['match_id']} pertence a {row['split']}; o executor exige {expected_split}."
                )
            continue
        video_path = videos_dir / row["video_filename"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        selected_rows.append(row)
    if not selected_rows:
        raise ValueError(f"Nenhuma partida {expected_split.upper()} foi selecionada.")
    return selected_rows


def predictions(result) -> list[tuple[tuple[float, float, float, float], float]]:
    return [
        (tuple(float(value) for value in box), float(confidence))
        for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
    ]


def collect_hits(
    video: dict[str, str], videos_dir: Path, goal_model: YOLO, ball_model: YOLO, args: argparse.Namespace,
    progress: Callable[[int, int], None] | None = None,
) -> list[Hit]:
    capture = cv2.VideoCapture(str(videos_dir / video["video_filename"]))
    if not capture.isOpened():
        raise FileNotFoundError(videos_dir / video["video_filename"])
    hits: list[Hit] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        step = max(1, round(fps / args.analysis_fps))
        report_interval = max(step, total_frames // 200)
        frame_index = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if progress and (frame_index % report_interval == 0 or frame_index + 1 >= total_frames):
                progress(min(frame_index + 1, total_frames), total_frames)
            if frame_index % step:
                continue
            goal_result = goal_model.predict(
                source=frame, conf=args.goal_confidence, imgsz=args.goal_imgsz,
                device=args.device, verbose=False,
            )[0]
            goal_predictions = predictions(goal_result)
            if not goal_predictions:
                continue
            ball_result = ball_model.predict(
                source=frame, conf=args.ball_confidence, imgsz=args.ball_imgsz,
                device=args.device, verbose=False,
            )[0]
            ball_predictions = predictions(ball_result)
            for ball_box, ball_confidence in ball_predictions:
                ball_center = box_center(ball_box)
                nearest_goal, goal_confidence = min(
                    goal_predictions,
                    key=lambda item: normalized_distance(ball_center, item[0]),
                )
                distance = normalized_distance(ball_center, nearest_goal)
                expanded = expand_box(nearest_goal, args.goal_margin)
                is_inside = inside(ball_center, expanded)
                if not is_inside and distance > args.max_normalized_distance:
                    continue
                proximity = max(0.0, 1.0 - distance / args.max_normalized_distance)
                score = ball_confidence * (1.0 + proximity + (1.0 if is_inside else 0.0))
                hits.append(Hit(
                    match_id=video["match_id"], video_filename=video["video_filename"],
                    timestamp=frame_index / fps, frame_index=frame_index, goal_box=nearest_goal,
                    goal_confidence=goal_confidence, ball_box=ball_box,
                    ball_confidence=ball_confidence, inside_expanded_goal=is_inside,
                    normalized_distance=distance, score=score,
                ))
    finally:
        capture.release()
    return hits


def group_hits(hits: list[Hit], merge_gap_seconds: float, minimum_hits: int) -> list[list[Hit]]:
    groups: list[list[Hit]] = []
    for hit in sorted(hits, key=lambda item: item.timestamp):
        if not groups or hit.timestamp - groups[-1][-1].timestamp > merge_gap_seconds:
            groups.append([hit])
        else:
            groups[-1].append(hit)
    return [group for group in groups if len(group) >= minimum_hits]


def select_diverse_groups(groups: list[list[Hit]], args: argparse.Namespace) -> list[list[Hit]]:
    selected: list[list[Hit]] = []
    for group in sorted(groups, key=lambda item: max(hit.score for hit in item), reverse=True):
        best_time = max(group, key=lambda item: item.score).timestamp
        if any(abs(best_time - max(other, key=lambda item: item.score).timestamp) < args.min_separation_seconds for other in selected):
            continue
        selected.append(group)
        if len(selected) >= args.max_candidates_per_match:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_progress(
    output: Path,
    status: str,
    processed_matches: list[str],
    summary_by_match: dict[str, dict[str, int]],
    candidate_events: int,
) -> None:
    progress = {
        "status": status,
        "processed_matches": processed_matches,
        "candidate_events_so_far": candidate_events,
        "by_match": dict(summary_by_match),
    }
    (output / "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(
    argv: list[str] | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Minera clipes candidatos a evento de gol exclusivamente em partidas TRAIN."
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--videos-dir", type=Path)
    parser.add_argument("--video-path", type=Path, help="Vídeo externo a analisar sem consultar o split.")
    parser.add_argument("--video-id", default="video", help="Identificador local para vídeo externo.")
    parser.add_argument("--goal-model", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-matches", nargs="+", default=None)
    parser.add_argument("--expected-split", choices=["train", "validation"], default="train")
    parser.add_argument("--analysis-fps", type=float, default=5.0)
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.05)
    parser.add_argument("--goal-imgsz", type=int, default=640)
    parser.add_argument("--ball-imgsz", type=int, default=960)
    parser.add_argument("--goal-margin", type=float, default=0.35)
    parser.add_argument("--max-normalized-distance", type=float, default=2.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-hits", type=int, default=2)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--max-candidates-per-match", type=int, default=4)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.analysis_fps <= 0 or args.minimum_hits < 1 or args.max_candidates_per_match < 1:
        raise ValueError("analysis-fps, minimum-hits e max-candidates-per-match devem ser positivos.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}. Escolha uma pasta nova.")
    if not args.goal_model.is_file() or not args.ball_model.is_file():
        raise FileNotFoundError("Os pesos informados para estrutura do gol e bola devem existir.")

    if args.video_path:
        if not args.video_path.is_file():
            raise FileNotFoundError(args.video_path)
        args.videos_dir = args.video_path.parent
        videos = [{"match_id": args.video_id, "video_filename": args.video_path.name, "split": "external"}]
    else:
        if not args.split_manifest or not args.videos_dir:
            raise ValueError("Informe --video-path ou --split-manifest com --videos-dir.")
        videos = load_videos(args.split_manifest, args.videos_dir, args.include_matches, args.expected_split)
    args.output.mkdir(parents=True, exist_ok=False)
    write_progress(args.output, "running", [], {}, 0)
    goal_model, ball_model = YOLO(str(args.goal_model)), YOLO(str(args.ball_model))
    all_frame_rows: list[dict[str, str]] = []
    all_event_rows: list[dict[str, str]] = []
    summary_by_match: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "groups": 0, "selected": 0})
    processed_matches: list[str] = []
    for video_index, video in enumerate(videos):
        print(f"Iniciando: {video['match_id']}", flush=True)
        def report_frames(current: int, total: int, index: int = video_index) -> None:
            if progress:
                completed = (index + current / total) / len(videos)
                progress("Mineração de candidatos", 70.0 * completed)

        hits = collect_hits(video, args.videos_dir, goal_model, ball_model, args, report_frames)
        groups = group_hits(hits, args.merge_gap_seconds, args.minimum_hits)
        selected = select_diverse_groups(groups, args)
        summary_by_match[video["match_id"]] = {"hits": len(hits), "groups": len(groups), "selected": len(selected)}
        for hit in hits:
            all_frame_rows.append({
                "match_id": hit.match_id, "video_filename": hit.video_filename,
                "timestamp_seconds": f"{hit.timestamp:.3f}", "frame_index": str(hit.frame_index),
                "goal_box_json": json.dumps(hit.goal_box), "goal_confidence": f"{hit.goal_confidence:.6f}",
                "ball_box_json": json.dumps(hit.ball_box), "ball_confidence": f"{hit.ball_confidence:.6f}",
                "ball_inside_expanded_goal": str(hit.inside_expanded_goal),
                "normalized_distance": f"{hit.normalized_distance:.6f}", "candidate_score": f"{hit.score:.6f}",
            })
        for group in selected:
            best = max(group, key=lambda item: item.score)
            event_id = f"goal_candidate_v1_{len(all_event_rows) + 1:04d}"
            start = max(0.0, best.timestamp - args.clip_pre_seconds)
            end = best.timestamp + args.clip_post_seconds
            all_event_rows.append({
                "event_id": event_id,
                "source_file": "mine_goal_event_candidates_v1.py",
                "source_interval_id": f"{video['match_id']}_{best.frame_index}",
                "source_interval_label": "bola_proxima_estrutura_gol",
                "match_id": video["match_id"], "video_filename": video["video_filename"], "split": args.expected_split,
                "event_timestamp_seconds": f"{best.timestamp:.3f}", "event_time_hhmmss": format_time(best.timestamp),
                "ground_truth": "", "ball_visible": "", "goal_visible": "", "spatial_candidate": "True",
                "relative_motion_valid": "", "identity_stable": "", "center_visible_after": "",
                "ball_at_center_after": "", "players_reset": "", "kickoff_observed": "", "restart_time": "",
                "scoreboard_visible": "", "score_changed": "", "not_evaluable_reason": "",
                "annotation_status": "pending",
                "notes": "Candidato minerado automaticamente; revisão humana obrigatória.",
                "candidate_score": f"{best.score:.6f}", "candidate_hit_frames": str(len(group)),
                "candidate_start_seconds": f"{start:.3f}", "candidate_end_seconds": f"{end:.3f}",
            })
        processed_matches.append(video["match_id"])
        write_csv(args.output / "candidate_events_v1.csv", all_event_rows, EVENT_FIELDS)
        write_csv(args.output / "candidate_frames_v1.csv", all_frame_rows, FRAME_FIELDS)
        write_progress(
            args.output, "running", processed_matches, summary_by_match, len(all_event_rows)
        )
        print(
            f"Concluída: {video['match_id']} | hits={len(hits)} grupos={len(groups)} selecionados={len(selected)}",
            flush=True,
        )

    write_csv(args.output / "candidate_events_v1.csv", all_event_rows, EVENT_FIELDS)
    write_csv(args.output / "candidate_frames_v1.csv", all_frame_rows, FRAME_FIELDS)
    metrics = {
        "experiment_id": args.output.name,
        "scope": f"somente {args.expected_split.upper()}",
        "videos_processed": len(videos),
        "candidate_events": len(all_event_rows),
        "parameters": vars(args) | {"split_manifest": str(args.split_manifest), "videos_dir": str(args.videos_dir), "goal_model": str(args.goal_model), "ball_model": str(args.ball_model), "output": str(args.output)},
        "by_match": summary_by_match,
        "warning": "Gerador de hipóteses; não mede desempenho final nem confirma gol.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_progress(args.output, "completed", processed_matches, summary_by_match, len(all_event_rows))
    print(json.dumps({"candidate_events": len(all_event_rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
