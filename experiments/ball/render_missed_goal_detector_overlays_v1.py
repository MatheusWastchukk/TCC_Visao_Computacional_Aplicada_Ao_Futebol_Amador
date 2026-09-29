"""Recria clipes TRAIN de misses com as caixas dos detectores congelados."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1"
GOAL_MODEL = ROOT / "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt"
BALL_MODEL = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
REVIEW_STATUSES = {"suppressed_by_nms", "outside_top60_after_nms"}


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return [
        row for row in rows
        if row.get("pool_recoverable", "").lower() == "true"
        and row.get("audit_status") in REVIEW_STATUSES
    ]


def draw_box(frame, box, confidence: float, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    text = f"{label} {confidence:.2f}"
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(0, y1 - height - baseline - 5)
    cv2.rectangle(frame, (x1, top), (x1 + width + 6, top + height + baseline + 5), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + height + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def render_case(row: dict[str, str], videos_dir: Path, output: Path, goal_model: YOLO, ball_model: YOLO,
                device: str, before: float, after: float, analysis_fps: float,
                goal_conf: float, ball_conf: float, goal_imgsz: int, ball_imgsz: int) -> dict[str, object]:
    video_path = videos_dir / row["video_filename"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    output_path = output / f"{row['goal_catalog_id']}.mp4"
    partial_path = output / f"{row['goal_catalog_id']}.partial.mp4"
    writer = None
    frames_read = frames_inferred = goal_positive_frames = 0
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / source_fps if frame_count else float(row["goal_timestamp_seconds"]) + after
        goal_time = float(row["goal_timestamp_seconds"])
        start_time = max(0.0, goal_time - before)
        end_time = min(duration, goal_time + after)
        start_frame = max(0, int(start_time * source_fps))
        end_frame = min(max(0, frame_count - 1), int(end_time * source_fps))
        step = max(1, round(source_fps / analysis_fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first = capture.read()
        if not ok or first is None:
            raise RuntimeError(f"Falha ao ler o clipe em {video_path} a partir de {start_time:.2f}s")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(partial_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Falha ao criar o video: {partial_path}")

        current_frame = start_frame
        cached_goal: list[tuple[list[float], float]] = []
        cached_ball: list[tuple[list[float], float]] = []
        ball_was_queried = False
        frame = first
        while ok and current_frame <= end_frame:
            if current_frame % step == 0:
                goal_result = goal_model.predict(source=frame, conf=goal_conf, imgsz=goal_imgsz, device=device, verbose=False)[0]
                cached_goal = [
                    (box.tolist(), float(confidence))
                    for box, confidence in zip(goal_result.boxes.xyxy.cpu(), goal_result.boxes.conf.cpu())
                ]
                ball_was_queried = bool(cached_goal)
                cached_ball = []
                if ball_was_queried:
                    ball_result = ball_model.predict(source=frame, conf=ball_conf, imgsz=ball_imgsz, device=device, verbose=False)[0]
                    cached_ball = [
                        (box.tolist(), float(confidence))
                        for box, confidence in zip(ball_result.boxes.xyxy.cpu(), ball_result.boxes.conf.cpu())
                    ]
                frames_inferred += 1
                goal_positive_frames += int(ball_was_queried)

            annotated = frame.copy()
            for box, confidence in cached_goal:
                draw_box(annotated, box, confidence, "GOAL", (255, 220, 0))
            for box, confidence in cached_ball:
                draw_box(annotated, box, confidence, "BALL", (0, 220, 255))
            if not ball_was_queried:
                ball_text = "BALL: detector nao consultado (nenhum GOAL detectado)"
            else:
                ball_text = f"BALL: {len(cached_ball)} deteccao(oes)"
            elapsed = current_frame / source_fps
            header = f"{row['goal_catalog_id']} | {row['match_id']} | video {elapsed:.1f}s | gol catalogado {goal_time:.1f}s"
            cv2.rectangle(annotated, (0, 0), (width, 69), (18, 18, 18), -1)
            cv2.putText(annotated, header, (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"GOAL: {len(cached_goal)} | {ball_text} | caixas inferidas a {analysis_fps:g} FPS; mantidas entre amostras", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(annotated)
            frames_read += 1
            current_frame += 1
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not frames_read:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Nenhum quadro gerado para {row['goal_catalog_id']}")
    partial_path.replace(output_path)
    return {
        "goal_catalog_id": row["goal_catalog_id"], "match_id": row["match_id"],
        "video_filename": row["video_filename"], "goal_timestamp_seconds": goal_time,
        "clip_start_seconds": start_time, "clip_end_seconds": end_time,
        "output_video": str(output_path.relative_to(ROOT)), "source_fps": source_fps,
        "output_frames": frames_read, "inference_frames": frames_inferred,
        "frames_with_goal_detection": goal_positive_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=AUDIT_DIR / "train_missed_goal_audit_v1.csv")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--output", type=Path, default=AUDIT_DIR / "overlay_clips_v1")
    parser.add_argument("--goal-model", type=Path, default=GOAL_MODEL)
    parser.add_argument("--ball-model", type=Path, default=BALL_MODEL)
    parser.add_argument("--device", default="0", help="0 para GPU ROCm ou cpu como fallback.")
    parser.add_argument("--before", type=float, default=25.0)
    parser.add_argument("--after", type=float, default=35.0)
    parser.add_argument("--analysis-fps", type=float, default=5.0)
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.05)
    parser.add_argument("--goal-imgsz", type=int, default=640)
    parser.add_argument("--ball-imgsz", type=int, default=960)
    args = parser.parse_args()
    for path in (args.manifest, args.goal_model, args.ball_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.manifest)
    goal_model, ball_model = YOLO(str(args.goal_model)), YOLO(str(args.ball_model))
    actual_device = "CPU"
    if args.device != "cpu" and torch.cuda.is_available():
        actual_device = f"{torch.cuda.get_device_name(0)} (ROCm/HIP {torch.version.hip})"
    done_path = args.output / "overlay_manifest_v1.json"
    existing = {}
    if done_path.is_file():
        existing = {row["goal_catalog_id"]: row for row in json.loads(done_path.read_text(encoding="utf-8"))["clips"]}
    clips = dict(existing)
    for index, row in enumerate(cases, 1):
        key = row["goal_catalog_id"]
        target = args.output / f"{key}.mp4"
        if target.is_file() and key in clips:
            print(f"[{index}/{len(cases)}] ja concluido: {key}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] inferindo e renderizando: {key} | device={args.device}", flush=True)
        clip = render_case(row, args.videos_dir, args.output, goal_model, ball_model, args.device, args.before,
                           args.after, args.analysis_fps, args.goal_confidence, args.ball_confidence,
                           args.goal_imgsz, args.ball_imgsz)
        clips[key] = clip
        done_path.write_text(json.dumps({
            "experiment_id": "GOAL-EVENT-MISSED-GOAL-OVERLAY-TRAIN-V1",
            "scope": "TRAIN only; boxes recomputed with frozen detector weights and candidate-mining thresholds; TEST closed",
            "device_requested": args.device, "device_actual": actual_device,
            "torch_version": torch.__version__, "goal_model": str(args.goal_model), "ball_model": str(args.ball_model),
            "analysis_fps": args.analysis_fps, "goal_confidence": args.goal_confidence,
            "ball_confidence": args.ball_confidence, "goal_imgsz": args.goal_imgsz,
            "ball_imgsz": args.ball_imgsz, "clip_before_seconds": args.before,
            "clip_after_seconds": args.after, "clips": list(clips.values()),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(cases)}] concluido: {key} | inferencias={clip['inference_frames']} | saida={target}", flush=True)
    done_path.write_text(json.dumps({
        "experiment_id": "GOAL-EVENT-MISSED-GOAL-OVERLAY-TRAIN-V1",
        "scope": "TRAIN only; boxes recomputed with frozen detector weights and candidate-mining thresholds; TEST closed",
        "device_requested": args.device, "device_actual": actual_device,
        "torch_version": torch.__version__, "goal_model": str(args.goal_model), "ball_model": str(args.ball_model),
        "analysis_fps": args.analysis_fps, "goal_confidence": args.goal_confidence,
        "ball_confidence": args.ball_confidence, "goal_imgsz": args.goal_imgsz,
        "ball_imgsz": args.ball_imgsz, "clip_before_seconds": args.before,
        "clip_after_seconds": args.after, "clips": list(clips.values()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Concluido: {len(clips)}/{len(cases)} clipes | device={actual_device} | {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
