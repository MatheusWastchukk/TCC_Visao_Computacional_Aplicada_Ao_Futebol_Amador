"""Materialize one V10 train/validation split for Ultralytics YOLO."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a V10 YOLO split.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-match", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    train_rows = [row for row in manifests if row["match_id"] != args.validation_match]
    validation_rows = [row for row in manifests if row["match_id"] == args.validation_match]
    if not train_rows or not validation_rows:
        raise ValueError("O split precisa conter treino e validação")
    for split, split_rows in [("train", train_rows), ("val", validation_rows)]:
        image_dir = args.output / "images" / split
        label_dir = args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for row in split_rows:
            source_image = Path(row["image_path"])
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            image_name = f"{row['frame_id']}.jpg"
            shutil.copy2(source_image, image_dir / image_name)
            shutil.copy2(row["label_path"], label_dir / f"{row['frame_id']}.txt")
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve()}\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: [goal_structure]\n",
        encoding="utf-8",
    )
    print(
        {
            "train_frames": len(train_rows),
            "validation_frames": len(validation_rows),
            "validation_match": args.validation_match,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
