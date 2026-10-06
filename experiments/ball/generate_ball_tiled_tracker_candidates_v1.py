"""Generate label-blind global+tile BALL candidate lists for frozen tracker inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tmp/ultralytics_tiled_tracker_config"))
DEFAULT_INPUT = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/blind_tracker_inputs_full_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001"
CHECKPOINTS = {
    "baseline": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "oversampled": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
}
CONFIDENCE = 0.01
MAX_PER_VIEW = 300
MAX_SELECTED = 10
IMG_SIZE = 960
OVERLAP = 0.20
DEDUP_IOU = 0.75
OUTPUT_NAME = "tracker_candidates_blind_v1.csv"
FIELDS = ("checkpoint", "sequence_id", "sequence_frame", "sample_id", "rank", "confidence", "x1", "y1", "x2", "y2", "image_width", "image_height", "global_proposals", "tiled_proposals", "pooled_proposals", "deduplicated_proposals")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows file scanners can briefly hold the previous progress snapshot.
        # The progress file is recoverable from the flushed prediction CSV.
        path.write_text(payload, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            pass


def box_iou(a, b) -> float:
    iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def tiles(frame):
    height, width = frame.shape[:2]
    tile_w, tile_h = min(width, math.ceil(width / (2 - OVERLAP))), min(height, math.ceil(height / (2 - OVERLAP)))
    for row_index, y in enumerate((0, height - tile_h)):
        for col_index, x in enumerate((0, width - tile_w)):
            yield frame[y:y + tile_h, x:x + tile_w], x, y


def predict(model: YOLO, frame, device: str, offset_x: int = 0, offset_y: int = 0):
    result = model.predict(source=frame, conf=CONFIDENCE, imgsz=IMG_SIZE, device=device,
                           max_det=MAX_PER_VIEW, verbose=False)[0]
    if result.boxes is None:
        return []
    return [
        {"box": (float(b[0]) + offset_x, float(b[1]) + offset_y, float(b[2]) + offset_x, float(b[3]) + offset_y),
         "confidence": float(conf)}
        for b, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True)
    ]


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept = []
    for row in sorted(rows, key=lambda r: -float(r["confidence"])):
        if all(box_iou(row["box"], other["box"]) < DEDUP_IOU for other in kept):
            kept.append(row)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    input_rows = list(csv.DictReader(args.input.open(newline="", encoding="utf-8-sig")))
    required = {"sequence_id", "sequence_frame", "sample_id", "image_path", "match_id"}
    if not input_rows or not required <= set(input_rows[0]):
        raise ValueError("Blind tracker input has an invalid schema")
    if len(input_rows) != 1435 or len({row["sequence_id"] for row in input_rows}) != 35:
        raise ValueError("Expected 35 frozen sequences and 1,435 blind input frames")
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / OUTPUT_NAME
    progress_path = args.output / "candidate_generation_progress_v1.json"
    runtime_path = args.output / "candidate_generation_runtime_v1.json"
    config = {
        "input_path": str(args.input.relative_to(ROOT)), "input_sha256": sha256(args.input),
        "checkpoint_hashes": {name: sha256(path) for name, path in CHECKPOINTS.items()},
        "confidence": CONFIDENCE, "imgsz": IMG_SIZE, "max_det_per_view": MAX_PER_VIEW,
        "tile_grid": "2x2", "overlap": OVERLAP, "cross_view_dedup_iou": DEDUP_IOU,
        "max_selected_candidates": MAX_SELECTED, "device": args.device,
    }
    if runtime_path.exists():
        prior = json.loads(runtime_path.read_text(encoding="utf-8"))
        if prior != config:
            raise ValueError("Existing runtime metadata differs; refusing to resume with changed inputs/settings")
    else:
        if predictions_path.exists() and predictions_path.stat().st_size:
            raise FileExistsError(f"Prediction CSV exists without matching runtime metadata: {predictions_path}")
        atomic_json(runtime_path, config)
    completed: set[str] = set()
    if progress_path.exists():
        completed = set(json.loads(progress_path.read_text(encoding="utf-8")).get("completed", []))
    if predictions_path.is_file() and predictions_path.stat().st_size:
        with predictions_path.open(newline="", encoding="utf-8-sig") as stream:
            for saved in csv.DictReader(stream):
                completed.add(f"{saved['checkpoint']}|{saved['sample_id']}")
    mode = "a" if predictions_path.exists() and predictions_path.stat().st_size else "w"
    with predictions_path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
            model = YOLO(str(checkpoint_path))
            warm = cv2.imread(input_rows[0]["image_path"])
            predict(model, warm, args.device)
            model_completed = 0
            for row in input_rows:
                key = f"{checkpoint_name}|{row['sample_id']}"
                if key in completed:
                    continue
                image = cv2.imread(row["image_path"])
                if image is None:
                    raise FileNotFoundError(row["image_path"])
                height, width = image.shape[:2]
                global_rows = predict(model, image, args.device)
                tile_rows = []
                for crop, x, y in tiles(image):
                    tile_rows.extend(predict(model, crop, args.device, x, y))
                pooled = [*global_rows, *tile_rows]
                unique = deduplicate(pooled)
                chosen = unique[:MAX_SELECTED]
                for rank, item in enumerate(chosen, 1):
                    x1, y1, x2, y2 = item["box"]
                    writer.writerow({
                        "checkpoint": checkpoint_name, "sequence_id": row["sequence_id"],
                        "sequence_frame": row["sequence_frame"], "sample_id": row["sample_id"],
                        "rank": rank, "confidence": f"{float(item['confidence']):.8f}",
                        "x1": f"{x1:.3f}", "y1": f"{y1:.3f}", "x2": f"{x2:.3f}", "y2": f"{y2:.3f}",
                        "image_width": width, "image_height": height,
                        "global_proposals": len(global_rows), "tiled_proposals": len(tile_rows),
                        "pooled_proposals": len(pooled), "deduplicated_proposals": len(unique),
                    })
                stream.flush()
                completed.add(key)
                model_completed += 1
                atomic_json(progress_path, {"completed": sorted(completed), "completed_count": len(completed), "expected_count": len(CHECKPOINTS) * len(input_rows)})
                if model_completed % 50 == 0 or model_completed == len(input_rows):
                    print(f"{checkpoint_name}: {model_completed}/{len(input_rows)} frames", flush=True)
            del model
    expected = len(CHECKPOINTS) * len(input_rows)
    if len(completed) != expected:
        raise RuntimeError(f"Incomplete tiled predictions: {len(completed)}/{expected}")
    print(json.dumps({"status": "PREDICTION_COMPLETE", "rows": len(completed), "csv": str(predictions_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
