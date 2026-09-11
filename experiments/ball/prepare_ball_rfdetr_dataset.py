"""Prepara uma cópia COCO do dataset anotado da bola para o RF-DETR."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara dataset COCO para o RF-DETR.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows = [row for row in rows if row["ball_visibility"] in {"VISIBLE", "PARTIAL", "NOT_VISIBLE"}]
    if not rows:
        raise ValueError("Nenhuma anotação utilizável foi encontrada.")

    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        split_dir = args.output / ("valid" if split == "validation" else split)
        image_dir = split_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        coco_images: list[dict[str, object]] = []
        coco_annotations: list[dict[str, object]] = []
        for image_id, row in enumerate(split_rows, start=1):
            source = Path(row["image_path"])
            target = image_dir / f"{row['sample_id']}.jpg"
            shutil.copy2(source, target)
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Não foi possível abrir a imagem: {source}")
            height, width = image.shape[:2]
            coco_images.append({
                "id": image_id,
                "file_name": f"images/{target.name}",
                "width": width,
                "height": height,
                "sample_id": row["sample_id"],
                "match_id": row["match_id"],
                "ball_visibility": row["ball_visibility"],
            })
            if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
                x1, y1, x2, y2 = [float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")]
                coco_annotations.append({
                    "id": len(coco_annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                })
        payload = {
            "info": {"description": "Cópia COCO do dataset da bola do TCC", "version": "v2"},
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "ball", "supercategory": "ball"}],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {
        "source_annotations": str(args.annotations),
        "included_rows": len(rows),
        "train_images": sum(row["split"] == "train" for row in rows),
        "validation_images": sum(row["split"] == "validation" for row in rows),
        "positive_images": sum(row["ball_visibility"] in {"VISIBLE", "PARTIAL"} for row in rows),
        "not_visible_images": sum(row["ball_visibility"] == "NOT_VISIBLE" for row in rows),
        "category": "ball",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
