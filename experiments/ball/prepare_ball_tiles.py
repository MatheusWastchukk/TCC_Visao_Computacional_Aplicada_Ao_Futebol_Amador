"""Prepara um dataset YOLO 2x2 para aumentar a escala aparente da bola."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Divide os frames da bola em tiles 2x2.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tile_label(lines: list[str], left: int, top: int, width: int, height: int) -> list[str]:
    result: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        ix1, iy1 = max(x1, left), max(y1, top)
        ix2, iy2 = min(x2, left + width // 2), min(y2, top + height // 2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        tile_w, tile_h = width // 2, height // 2
        tx1, ty1, tx2, ty2 = ix1 - left, iy1 - top, ix2 - left, iy2 - top
        result.append(f"0 {((tx1 + tx2) / 2) / tile_w:.8f} {((ty1 + ty2) / 2) / tile_h:.8f} {(tx2 - tx1) / tile_w:.8f} {(ty2 - ty1) / tile_h:.8f}")
    return result


def main() -> int:
    args = parse_args()
    with (args.input / "manifest.csv").open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    output_rows: list[dict[str, str]] = []
    for row in manifest:
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        height, width = image.shape[:2]
        source_lines = Path(row["label_path"]).read_text(encoding="utf-8").splitlines()
        for tile_id, (left, top) in enumerate(((0, 0), (width // 2, 0), (0, height // 2), (width // 2, height // 2))):
            tile = image[top : top + height // 2, left : left + width // 2]
            sample_id = f"{row['sample_id']}_tile{tile_id}"
            split = row["split"]
            image_target = args.output / "images" / split / f"{sample_id}.jpg"
            label_target = args.output / "labels" / split / f"{sample_id}.txt"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_target), tile)
            label_target.write_text("\n".join(tile_label(source_lines, left, top, width, height)) + "\n", encoding="utf-8")
            output_rows.append({"sample_id": sample_id, "match_id": row["match_id"], "split": split, "parent_sample_id": row["sample_id"], "image_path": image_target.as_posix(), "label_path": label_target.as_posix()})

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"tiles={len(output_rows)}")
    print(f"train={sum(row['split'] == 'train' for row in output_rows)}")
    print(f"validation={sum(row['split'] == 'validation' for row in output_rows)}")
    print(f"positivos={sum(bool(Path(row['label_path']).read_text(encoding='utf-8').strip()) for row in output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
