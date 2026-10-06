"""Build TRAIN-only context/zoom sheets for the 127 positive BALL misses.

This creates visual review aids only. It does not assign cause tags or change
the canonical annotation CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent


def fit(image, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_card(row: dict[str, str], image: np.ndarray, box: list[float]) -> np.ndarray:
    card = np.full((340, 660, 3), 24, dtype=np.uint8)
    x1, y1, x2, y2 = map(float, box)
    h, w = image.shape[:2]
    cv2.putText(
        card,
        f"{row['sequence_id']} frame {row['sequence_frame']} | {row['ball_visibility']}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        card,
        f"candidate_count={row['candidate_count']} | manual GT box in yellow | no cause label",
        (8, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    context = fit(image, 310, 286)
    context_scale = min(310 / w, 286 / h)
    pad_x = (310 - round(w * context_scale)) // 2
    pad_y = (286 - round(h * context_scale)) // 2
    p1 = (pad_x + round(x1 * context_scale), pad_y + round(y1 * context_scale))
    p2 = (pad_x + round(x2 * context_scale), pad_y + round(y2 * context_scale))
    cv2.rectangle(context, p1, p2, (0, 255, 255), 2)
    card[48:334, 8:318] = context

    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    side = int(max(160, bw * 8, bh * 8))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = left + side, top + side
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(w, right), min(h, bottom)
    patch = image[src_top:src_bottom, src_left:src_right]
    patch = cv2.copyMakeBorder(
        patch,
        max(0, -top),
        max(0, bottom - h),
        max(0, -left),
        max(0, right - w),
        cv2.BORDER_REPLICATE,
    )
    zoom = cv2.resize(patch, (310, 286), interpolation=cv2.INTER_CUBIC)
    zx1 = round((x1 - left) * 310 / side)
    zy1 = round((y1 - top) * 286 / side)
    zx2 = round((x2 - left) * 310 / side)
    zy2 = round((y2 - top) * 286 / side)
    cv2.rectangle(zoom, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
    card[48:334, 334:644] = zoom
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXP / "positive_miss_mechanism_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=EXP / "visual_review_zoom_sheets_v1")
    parser.add_argument("--per-sheet", type=int, default=12)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig", newline="")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[np.ndarray] = []
    index: list[dict[str, str | int]] = []
    for row in rows:
        image_path = ROOT / Path(row["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        box = json.loads(row["ball_of_play_box_json"])
        cards.append(make_card(row, image, box))
        index.append(
            {
                "sequence_id": row["sequence_id"],
                "match_id": row["match_id"],
                "sequence_frame": int(row["sequence_frame"]),
                "image_path": row["image_path"],
                "card_index": len(cards),
                "candidate_count": int(row["candidate_count"]),
            }
        )

    per_sheet = max(1, args.per_sheet)
    for start in range(0, len(cards), per_sheet):
        batch = cards[start : start + per_sheet]
        page = np.full((4 * 340, 3 * 660, 3), 12, dtype=np.uint8)
        for offset, card in enumerate(batch):
            row_index, col_index = divmod(offset, 3)
            y, x = row_index * 340, col_index * 660
            page[y : y + 340, x : x + 660] = card
        number = start // per_sheet + 1
        target = args.output_dir / f"contact_sheet_{number:02d}.jpg"
        if not cv2.imwrite(str(target), page, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Could not write {target}")

    (args.output_dir / "index_v1.json").write_text(
        json.dumps(
            {
                "source_csv": str(args.input),
                "scope": "127 frozen TRAIN misses; visualization only; no cause labels assigned",
                "rows": index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Created {len(cards)} visual cards in {args.output_dir}; labels unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
