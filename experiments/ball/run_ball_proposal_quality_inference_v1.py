"""Roda os dois detectores congelados nos 400 frames TRAIN da auditoria, GPU primeiro."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIOPEN_CACHE_ROOT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/miopen_workspace_cache_v1"
MIOPEN_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
for cache_subdir in ("user_db", "kernel_cache"):
    (MIOPEN_CACHE_ROOT / cache_subdir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(MIOPEN_CACHE_ROOT / "user_db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(MIOPEN_CACHE_ROOT / "kernel_cache"))

import cv2
import torch
from ultralytics import YOLO

DEFAULT_RUN = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2"
GOAL_MODEL = ROOT / "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt"
BALL_MODEL = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
FIELDS = ["sample_id", "match_id", "source_stratum", "detector", "rank_by_confidence", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict(model: YOLO, frame, confidence: float, imgsz: int):
    result = model.predict(source=frame, conf=confidence, imgsz=imgsz, device=0, verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        coords = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()
        classes = result.boxes.cls.tolist()
        boxes = sorted(zip(coords, scores, classes, strict=True), key=lambda x: -float(x[1]))
    return boxes


def draw_preview(frame, goals, balls):
    image = frame.copy()
    for rank, (box, conf, cls) in enumerate(goals, 1):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(image, f"GOAL {rank} {conf:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    palette = [(0, 0, 255), (255, 80, 0), (255, 0, 220), (0, 180, 0), (180, 80, 180), (0, 150, 255)]
    for rank, (box, conf, cls) in enumerate(balls, 1):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = palette[(rank - 1) % len(palette)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"BALL #{rank} {conf:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--goal-model", type=Path, default=GOAL_MODEL)
    parser.add_argument("--ball-model", type=Path, default=BALL_MODEL)
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.05)
    parser.add_argument("--goal-imgsz", type=int, default=640)
    parser.add_argument("--ball-imgsz", type=int, default=960)
    parser.add_argument("--device", choices=["0"], default="0")
    args = parser.parse_args()
    manifest = args.manifest or args.run_dir / "sample_manifest_v1.csv"
    samples = list(csv.DictReader(manifest.open(newline="", encoding="utf-8-sig")))
    sample_summary = json.loads((args.run_dir / "sample_summary_v1.json").read_text(encoding="utf-8"))
    expected_count = int(sample_summary["sample_count"])
    if len(samples) != expected_count or any(r["split"] != "train" for r in samples):
        raise ValueError(f"Expected exactly {expected_count} TRAIN samples; TEST is never accepted.")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm smoke failed: torch.cuda.is_available() is false; use the project CPU fallback environment.")
    if args.run_dir.exists() and not manifest.is_file():
        raise FileNotFoundError(manifest)
    for p in (args.goal_model, args.ball_model):
        if not p.is_file():
            raise FileNotFoundError(p)
    frames_dir, overlays_dir = args.run_dir / "frames_v1", args.run_dir / "overlays_v1"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    detection_csv = args.run_dir / "raw_detections_v1.csv"
    progress_path = args.run_dir / "inference_progress_v1.json"
    completed = set()
    if progress_path.exists():
        completed = set(json.loads(progress_path.read_text(encoding="utf-8")).get("completed_sample_ids", []))
    runtime_path = args.run_dir / "inference_runtime_v1.json"
    if not runtime_path.exists():
        atomic_json(runtime_path, {"backend": "ROCm/PyTorch", "device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "goal_model": str(args.goal_model), "ball_model": str(args.ball_model),
            "goal_confidence": args.goal_confidence, "ball_confidence": args.ball_confidence,
            "goal_imgsz": args.goal_imgsz, "ball_imgsz": args.ball_imgsz,
            "sampling": "fixed TRAIN-only manifest; every sampled frame is scored even if no GOAL or BALL proposal exists"})
    goal_model, ball_model = YOLO(str(args.goal_model)), YOLO(str(args.ball_model))
    mode = "a" if detection_csv.exists() else "w"
    with detection_csv.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, sample in enumerate(samples, 1):
            sid = sample["sample_id"]
            if sid in completed:
                continue
            video = args.videos_dir / sample["video_filename"]
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise FileNotFoundError(video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_index"]))
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode {video} frame {sample['frame_index']}.")
            goals = predict(goal_model, frame, args.goal_confidence, args.goal_imgsz)
            balls = predict(ball_model, frame, args.ball_confidence, args.ball_imgsz)
            for detector, detections in (("GOAL", goals), ("BALL", balls)):
                for rank, (box, conf, cls) in enumerate(detections, 1):
                    writer.writerow({"sample_id": sid, "match_id": sample["match_id"], "source_stratum": sample["source_stratum"],
                        "detector": detector, "rank_by_confidence": rank, "class_id": int(cls), "confidence": f"{conf:.7f}",
                        **{name: f"{float(value):.3f}" for name, value in zip(("x1", "y1", "x2", "y2"), box, strict=True)}})
            stream.flush()
            frame_name = f"{sid}.jpg"
            if not cv2.imwrite(str(frames_dir / frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not save source frame for {sid}.")
            preview = draw_preview(frame, goals, balls)
            if not cv2.imwrite(str(overlays_dir / frame_name), preview, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"Could not save detector overlay for {sid}.")
            completed.add(sid)
            atomic_json(progress_path, {"status": "running" if len(completed) < len(samples) else "completed",
                "completed_frames": len(completed), "total_frames": len(samples), "completed_sample_ids": sorted(completed)})
            if len(completed) % 20 == 0 or len(completed) == len(samples):
                print(f"{len(completed)}/{len(samples)} frames", flush=True)
    all_detections = list(csv.DictReader(detection_csv.open(newline="", encoding="utf-8-sig")))
    ball_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for det in all_detections:
        if det["detector"] == "BALL":
            ball_by_sample[det["sample_id"]].append(det)
    count_rows = []
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        boxes = ball_by_sample[sample["sample_id"]]
        count = len(boxes)
        by_stratum[sample["source_stratum"]].append(count)
        count_rows.append({"sample_id": sample["sample_id"], "match_id": sample["match_id"],
            "source_stratum": sample["source_stratum"], "sequence_id": sample.get("sequence_id", ""),
            "ball_proposal_count": count,
            "max_ball_confidence": f"{max((float(d['confidence']) for d in boxes), default=0.0):.7f}"})
    write_csv(args.run_dir / "proposal_counts_by_frame_v1.csv", count_rows)
    by_stratum_summary = {name: {"frames": len(counts), "frames_no_ball_proposal": sum(c == 0 for c in counts),
        "mean_proposals_per_frame": round(sum(counts)/len(counts), 6),
        "median_proposals_per_frame": sorted(counts)[len(counts)//2], "frames_5plus": sum(c >= 5 for c in counts)}
        for name, counts in by_stratum.items()}
    atomic_json(args.run_dir / "inference_summary_v1.json", {"frames": len(samples), "detections": len(all_detections),
        "ball_proposals_by_stratum": by_stratum_summary,
        "runtime": json.loads(runtime_path.read_text(encoding="utf-8")),
        "progress": {"status": "completed", "completed_frames": len(completed), "total_frames": len(samples)}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
