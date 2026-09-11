"""Diagnostica sinais visuais em janelas TRAIN centradas nos timestamps de gol."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


FIELDS = [
    "goal_catalog_id", "match_id", "video_filename", "split", "goal_timestamp_seconds",
    "window_start_seconds", "window_end_seconds", "frames_sampled", "frames_with_goal_detection",
    "frames_with_ball_detection", "frames_with_both_detections", "geometric_hit_frames",
    "automatic_failure_category", "annotation_status", "notes",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def normalized_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    width, height = box[2] - box[0], box[3] - box[1]
    diagonal = max(1.0, math.hypot(width, height))
    goal_center = center(box)
    return math.hypot(point[0] - goal_center[0], point[1] - goal_center[1]) / diagonal


def inside_expanded_goal(
    point: tuple[float, float], box: tuple[float, float, float, float], margin: float
) -> bool:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return x1 - width * margin <= point[0] <= x2 + width * margin and y1 - height * margin <= point[1] <= y2 + height * margin


def boxes(result) -> list[tuple[tuple[float, float, float, float], float]]:
    return [
        (tuple(float(value) for value in box), float(confidence))
        for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
    ]


def category(goal_frames: int, ball_frames: int, both_frames: int, geometric_frames: int) -> str:
    if not goal_frames and not ball_frames:
        return "SEM_ESTRUTURA_E_SEM_BOLA"
    if not goal_frames:
        return "SEM_ESTRUTURA_GOL"
    if not ball_frames:
        return "SEM_BOLA"
    if not both_frames:
        return "SEM_COOCCORRENCIA"
    if not geometric_frames:
        return "SEM_HIT_GEOMETRICO"
    return "HIT_GEOMETRICO"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostica sinais dos detectores em janelas de gols TRAIN conhecidos."
    )
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--goal-model", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-fps", type=float, default=5.0)
    parser.add_argument("--window-pre-seconds", type=float, default=10.0)
    parser.add_argument("--window-post-seconds", type=float, default=10.0)
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.05)
    parser.add_argument("--goal-imgsz", type=int, default=640)
    parser.add_argument("--ball-imgsz", type=int, default=960)
    parser.add_argument("--goal-margin", type=float, default=0.35)
    parser.add_argument("--max-normalized-distance", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-goals", type=int, default=None)
    args = parser.parse_args()
    if args.analysis_fps <= 0 or args.window_pre_seconds < 0 or args.window_post_seconds < 0:
        raise ValueError("analysis-fps e janelas devem ser positivos ou zero.")
    if not args.goal_model.is_file() or not args.ball_model.is_file():
        raise FileNotFoundError("Os pesos dos detectores devem existir.")

    goals, goal_fields = read_csv(args.goal_timestamps)
    required_goal_fields = {"goal_catalog_id", "match_id", "split", "event_timestamp_seconds", "annotation_status"}
    if missing := required_goal_fields - set(goal_fields):
        raise ValueError(f"Timestamps sem colunas obrigatórias: {sorted(missing)}")
    goals = [
        row for row in goals
        if row["split"] == "train" and row["annotation_status"] == "completed" and row["event_timestamp_seconds"]
    ]
    if args.max_goals is not None:
        goals = goals[:args.max_goals]
    if not goals:
        raise ValueError("Não há gols TRAIN avaliáveis selecionados.")

    split_rows, split_fields = read_csv(args.split_manifest)
    if missing := {"match_id", "video_filename", "split"} - set(split_fields):
        raise ValueError(f"Manifesto de split sem colunas: {sorted(missing)}")
    video_by_match = {row["match_id"]: row["video_filename"] for row in split_rows if row["split"] == "train"}
    unknown = sorted({row["match_id"] for row in goals} - set(video_by_match))
    if unknown:
        raise AssertionError(f"Gols fora de TRAIN ou sem vídeo: {unknown}")

    args.output.mkdir(parents=True, exist_ok=False)
    goal_model = YOLO(str(args.goal_model))
    ball_model = YOLO(str(args.ball_model))
    rows: list[dict[str, str]] = []
    for index, goal in enumerate(goals, start=1):
        timestamp = float(goal["event_timestamp_seconds"])
        video_filename = video_by_match[goal["match_id"]]
        capture = cv2.VideoCapture(str(args.videos_dir / video_filename))
        if not capture.isOpened():
            raise FileNotFoundError(args.videos_dir / video_filename)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            start = max(0.0, timestamp - args.window_pre_seconds)
            end = min(frame_count / fps, timestamp + args.window_post_seconds)
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
            step = max(1, round(fps / args.analysis_fps))
            frames_sampled = goal_frames = ball_frames = both_frames = geometric_frames = 0
            frame_index = round(start * fps) - 1
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                current_seconds = frame_index / fps
                if current_seconds > end:
                    break
                if frame_index % step:
                    continue
                frames_sampled += 1
                goal_boxes = boxes(goal_model.predict(
                    source=frame, conf=args.goal_confidence, imgsz=args.goal_imgsz,
                    device=args.device, verbose=False,
                )[0])
                ball_boxes = boxes(ball_model.predict(
                    source=frame, conf=args.ball_confidence, imgsz=args.ball_imgsz,
                    device=args.device, verbose=False,
                )[0])
                goal_frames += int(bool(goal_boxes))
                ball_frames += int(bool(ball_boxes))
                both_frames += int(bool(goal_boxes and ball_boxes))
                has_geometric_hit = False
                for ball_box, _ in ball_boxes:
                    point = center(ball_box)
                    for goal_box, _ in goal_boxes:
                        if inside_expanded_goal(point, goal_box, args.goal_margin) or normalized_distance(point, goal_box) <= args.max_normalized_distance:
                            has_geometric_hit = True
                            break
                    if has_geometric_hit:
                        break
                geometric_frames += int(has_geometric_hit)
        finally:
            capture.release()
        rows.append(
            {
                "goal_catalog_id": goal["goal_catalog_id"],
                "match_id": goal["match_id"],
                "video_filename": video_filename,
                "split": "train",
                "goal_timestamp_seconds": f"{timestamp:.3f}",
                "window_start_seconds": f"{start:.3f}",
                "window_end_seconds": f"{end:.3f}",
                "frames_sampled": str(frames_sampled),
                "frames_with_goal_detection": str(goal_frames),
                "frames_with_ball_detection": str(ball_frames),
                "frames_with_both_detections": str(both_frames),
                "geometric_hit_frames": str(geometric_frames),
                "automatic_failure_category": category(goal_frames, ball_frames, both_frames, geometric_frames),
                "annotation_status": "completed",
                "notes": "Diagnóstico automático em janela temporal de gol TRAIN; não confirma evento online.",
            }
        )
        write_csv(args.output / "goal_window_diagnosis_v1.csv", rows)
        progress = {"status": "running", "processed_goals": len(rows), "total_goals": len(goals)}
        (args.output / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Concluído {index}/{len(goals)}: {goal['goal_catalog_id']}", flush=True)
    counts = {}
    for row in rows:
        counts[row["automatic_failure_category"]] = counts.get(row["automatic_failure_category"], 0) + 1
    summary = {
        "experiment_id": "GOAL-EVENT-TIMESTAMP-WINDOW-DIAGNOSIS-V1",
        "scope": "somente TRAIN",
        "goals_diagnosed": len(rows),
        "automatic_failure_categories": dict(sorted(counts.items())),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "note": "Diagnóstico por sinais dos detectores ao redor de timestamps catalogados; não é avaliação final nem confirmação online.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "progress.json").write_text(json.dumps({"status": "completed", "processed_goals": len(rows), "total_goals": len(goals)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
