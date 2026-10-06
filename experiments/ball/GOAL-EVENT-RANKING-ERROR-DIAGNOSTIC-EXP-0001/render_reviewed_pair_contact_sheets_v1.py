"""Cria folhas de contato com inferência GOAL/BALL nova para pares TRAIN revisados."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
GOAL_MODEL = ROOT / "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt"
BALL_MODEL = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
OFFSETS = (-10, -5, 0, 5, 10)


def detections(result):
    if result.boxes is None:
        return []
    return [(box.tolist(), float(conf)) for box, conf in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu())]


def draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, 0.48, 1)
    y = max(h + base + 5, y)
    cv2.rectangle(image, (x, y - h - base - 4), (x + w + 7, y + 3), color, -1)
    cv2.putText(image, text, (x + 3, y - base), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def infer_frame(video_path, timestamp, goal_model, ball_model, device, goal_conf, ball_conf, goal_imgsz, ball_imgsz):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_no = min(max(0, round(timestamp * fps)), max(0, frame_count - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Falha ao ler {video_path} em {timestamp:.2f}s")
    goals = detections(goal_model.predict(source=frame, conf=goal_conf, imgsz=goal_imgsz, device=device, verbose=False)[0])
    balls = detections(ball_model.predict(source=frame, conf=ball_conf, imgsz=ball_imgsz, device=device, verbose=False)[0]) if goals else []
    annotated = frame.copy()
    for box, conf in goals:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 220, 0), 2, cv2.LINE_AA)
        draw_label(annotated, f"GOAL {conf:.2f}", x1, max(22, y1 - 2), (255, 220, 0))
    for box, conf in balls:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
        draw_label(annotated, f"BALL {conf:.2f}", x1, max(22, y1 - 2), (0, 220, 255))
    # Reduz as imagens mantendo a geometria; a inferência ocorreu na resolução original.
    scale = min(640 / annotated.shape[1], 360 / annotated.shape[0])
    annotated = cv2.resize(annotated, (round(annotated.shape[1] * scale), round(annotated.shape[0] * scale)))
    canvas = cv2.copyMakeBorder(annotated, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(canvas, f"{timestamp:.1f}s | GOAL={len(goals)} BALL={len(balls)}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas, {"timestamp_seconds": round(timestamp, 3), "goal_detections": len(goals), "ball_detections": len(balls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=EXP / "pairwise_control_audit_v1_annotated.csv")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--output", type=Path, default=EXP / "reviewed_pair_contact_sheets_v2")
    parser.add_argument("--goal-model", type=Path, default=GOAL_MODEL)
    parser.add_argument("--ball-model", type=Path, default=BALL_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fallback-reason", default=None, help="Motivo observado para executar em CPU depois da tentativa de GPU.")
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.05)
    parser.add_argument("--goal-imgsz", type=int, default=640)
    parser.add_argument("--ball-imgsz", type=int, default=960)
    args = parser.parse_args()
    for path in (args.audit, args.goal_model, args.ball_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A pasta de saída não está vazia: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.audit.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("manual_status") == "COMPLETED" and r.get("manual_label") == "NO_GOAL"]
    goal_model, ball_model = YOLO(str(args.goal_model)), YOLO(str(args.ball_model))
    clips = []
    actual_device = f"{torch.cuda.get_device_name(0)} (ROCm/HIP {torch.version.hip})" if args.device != "cpu" and torch.cuda.is_available() else "CPU"
    for case_index, row in enumerate(rows, 1):
        video_path = args.videos_dir / row["video_filename"]
        panels = []
        details = []
        for section, center in (("NO_GOAL candidate", float(row["control_candidate_timestamp_seconds"])), ("catalog goal", float(row["goal_timestamp_seconds"]))):
            for offset in OFFSETS:
                panel, stats = infer_frame(video_path, max(0, center + offset), goal_model, ball_model, args.device,
                                           args.goal_confidence, args.ball_confidence, args.goal_imgsz, args.ball_imgsz)
                draw_label(panel, f"{section} | offset {offset:+d}s", 5, 50, (30, 30, 150) if section == "NO_GOAL candidate" else (40, 115, 40))
                panels.append(panel)
                details.append({"section": section, "offset_seconds": offset, **stats})
        row_h = max(p.shape[0] for p in panels)
        row_w = max(p.shape[1] for p in panels)
        top_row = cv2.hconcat([cv2.resize(p, (row_w, row_h)) for p in panels[:len(OFFSETS)]])
        bottom_row = cv2.hconcat([cv2.resize(p, (row_w, row_h)) for p in panels[len(OFFSETS):]])
        sheet = cv2.vconcat([top_row, bottom_row])
        out = args.output / f"{row['goal_catalog_id']}.jpg"
        if not cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"Falha ao salvar {out}")
        clips.append({"goal_catalog_id": row["goal_catalog_id"], "match_id": row["match_id"], "video_filename": row["video_filename"],
                      "control_timestamp_seconds": float(row["control_candidate_timestamp_seconds"]),
                      "control_rank_full_pool": int(row["control_rank_full_pool"]), "control_temporal_score_lomo": float(row["control_temporal_score_lomo"]),
                      "catalog_goal_timestamp_seconds": float(row["goal_timestamp_seconds"]), "image": out.name, "frames": details})
        print(f"[{case_index}/{len(rows)}] {row['goal_catalog_id']} | {out.name}", flush=True)
    frame_sections = {section: [frame for case in clips for frame in case["frames"] if frame["section"] == section]
                      for section in ("NO_GOAL candidate", "catalog goal")}
    aggregate = {section: {"frames": len(frames), "frames_with_goal": sum(f["goal_detections"] > 0 for f in frames),
                           "frames_with_ball": sum(f["ball_detections"] > 0 for f in frames),
                           "mean_goal_boxes_per_frame": round(sum(f["goal_detections"] for f in frames) / max(1, len(frames)), 4),
                           "mean_ball_boxes_per_frame": round(sum(f["ball_detections"] for f in frames) / max(1, len(frames)), 4)}
                 for section, frames in frame_sections.items()}
    metadata = {"experiment_id": "GOAL-EVENT-RANKING-REVIEWED-PAIR-OVERLAYS-V1", "scope": "TRAIN only; fresh frozen-model inference; diagnostic contact sheets, not continuous tracking",
                "device_requested": args.device, "device_actual": actual_device, "fallback_reason": args.fallback_reason,
                "torch_version": torch.__version__, "goal_model": str(args.goal_model.relative_to(ROOT)),
                "ball_model": str(args.ball_model.relative_to(ROOT)), "goal_confidence": args.goal_confidence, "ball_confidence": args.ball_confidence,
                "goal_imgsz": args.goal_imgsz, "ball_imgsz": args.ball_imgsz, "offsets_seconds": OFFSETS,
                "aggregate_detections_by_section": aggregate, "clips": clips}
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Concluído: {len(clips)} folhas | device={actual_device} | {args.output}")


if __name__ == "__main__":
    main()
