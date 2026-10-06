"""Render positive oracle frames missed by the frozen detector for visual audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


EXP = Path(__file__).resolve().parent
RUN = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "run_v1"
INPUT = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "oracle_tracker_input_frames_16_26.csv"
OUT = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "detector_miss_review_v1"
TILE_W, TILE_H = 480, 192
CONTEXT_W, CONTEXT_H = 288, 142
CROP_W, CROP_H = TILE_W - CONTEXT_W, CONTEXT_H
PER_PAGE = 20


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def render_tile(row: dict[str, str], result: dict[str, str]) -> np.ndarray:
    image = cv2.imread(row["image_path"])
    if image is None:
        raise FileNotFoundError(row["image_path"])
    height, width = image.shape[:2]
    target = tuple(float(x) for x in json.loads(row["ball_of_play_box_json"]))
    x1, y1, x2, y2 = target
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pred = json.loads(result["selected_box_json"]) if result.get("selected_box_json") else None

    context = image.copy()
    cv2.rectangle(context, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 255), 3)
    if pred:
        px1, py1, px2, py2 = map(round, pred)
        cv2.rectangle(context, (px1, py1), (px2, py2), (0, 0, 255), 3)
    context = fit(context, CONTEXT_W, CONTEXT_H)

    crop_radius = max(56, int(max(x2 - x1, y2 - y1) * 4))
    left, top = max(0, int(cx - crop_radius)), max(0, int(cy - crop_radius))
    right, bottom = min(width, int(cx + crop_radius)), min(height, int(cy + crop_radius))
    crop = image[top:bottom, left:right].copy()
    cv2.rectangle(crop, (max(0, round(x1 - left)), max(0, round(y1 - top))),
                  (min(crop.shape[1] - 1, round(x2 - left)), min(crop.shape[0] - 1, round(y2 - top))), (0, 255, 255), 2)
    if pred:
        px1, py1, px2, py2 = map(round, pred)
        cv2.rectangle(crop, (max(0, px1 - left), max(0, py1 - top)),
                      (min(crop.shape[1] - 1, px2 - left), min(crop.shape[0] - 1, py2 - top)), (0, 0, 255), 2)
    crop = fit(crop, CROP_W, CROP_H)

    tile = np.full((TILE_H, TILE_W, 3), 30, dtype=np.uint8)
    cv2.putText(tile, f"{row['sequence_id']} frame {row['sequence_frame']} {row['ball_visibility']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "GT amarelo | selecionada vermelha | sem candidata IoU>=0.5", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    tile[48:48 + CONTEXT_H, :CONTEXT_W] = context
    tile[48:48 + CROP_H, CONTEXT_W:] = crop
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (90, 90, 90), 1)
    return tile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    annotations = read_csv(args.input)
    results = read_csv(args.run / "frame_identity_results.csv")
    result_by_key = {(r["sequence_id"], r["sequence_frame"]): r for r in results}
    misses = [
        row for row in annotations
        if row["sequence_id"].endswith("_POS")
        and row["ball_visibility"] in {"VISIBLE", "PARTIAL"}
        and row.get("ball_of_play_box_json")
        and result_by_key[(row["sequence_id"], row["sequence_frame"])]["target_candidate_rank"] == "0"
    ]
    misses.sort(key=lambda row: (row["sequence_id"], int(row["sequence_frame"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["sequence_id", "match_id", "sequence_frame", "timestamp_hhmmss", "ball_visibility", "image_path", "ball_of_play_box_json", "selected_box_json", "candidate_count", "review_failure_reason", "review_notes"]
    queue: list[dict[str, str]] = []
    for row in misses:
        result = result_by_key[(row["sequence_id"], row["sequence_frame"])]
        queue.append({
            "sequence_id": row["sequence_id"], "match_id": row["match_id"], "sequence_frame": row["sequence_frame"],
            "timestamp_hhmmss": row["timestamp_hhmmss"], "ball_visibility": row["ball_visibility"], "image_path": row["image_path"],
            "ball_of_play_box_json": row["ball_of_play_box_json"], "selected_box_json": result["selected_box_json"],
            "candidate_count": result["candidate_count"], "review_failure_reason": "", "review_notes": "",
        })
    with (args.output_dir / "positive_detector_misses_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
    for page_index in range(math.ceil(len(misses) / PER_PAGE)):
        page_rows = misses[page_index * PER_PAGE:(page_index + 1) * PER_PAGE]
        sheet = np.full((5 * TILE_H, 4 * TILE_W, 3), 18, dtype=np.uint8)
        for slot, row in enumerate(page_rows):
            result = result_by_key[(row["sequence_id"], row["sequence_frame"])]
            tile = render_tile(row, result)
            y, x = divmod(slot, 4)
            sheet[y * TILE_H:(y + 1) * TILE_H, x * TILE_W:(x + 1) * TILE_W] = tile
        path = args.output_dir / f"contact_sheet_{page_index + 1:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"Could not write {path}")
    print(f"positive detector misses: {len(misses)}")
    print(f"pages: {math.ceil(len(misses) / PER_PAGE)}")
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
