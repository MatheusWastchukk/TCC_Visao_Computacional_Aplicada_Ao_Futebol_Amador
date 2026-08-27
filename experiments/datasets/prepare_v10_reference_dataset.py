"""Materializa o dataset V10 a partir das anotações train e validation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


def yolo_line(row: dict[str, str], width: int, height: int) -> str:
    points = [
        (float(row["lt_x"]), float(row["lt_y"])),
        (float(row["rt_x"]), float(row["rt_y"])),
        (float(row["lb_x"]), float(row["lb_y"])),
        (float(row["rb_x"]), float(row["rb_y"])),
    ]
    left = min(point[0] for point in points)
    right = max(point[0] for point in points)
    top = min(point[1] for point in points)
    bottom = max(point[1] for point in points)
    values = [
        ((left + right) / 2) / width,
        ((top + bottom) / 2) / height,
        (right - left) / width,
        (bottom - top) / height,
    ]
    if values[2] <= 0 or values[3] <= 0 or any(value < 0 or value > 1 for value in values):
        raise ValueError(f"Caixa inválida em {row['frame_id']}: {values}")
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["frame_id"] in seen:
                    raise ValueError(f"Frame duplicado: {row['frame_id']}")
                if row["annotation_status"] != "completed":
                    raise ValueError(f"Anotação pendente: {row['frame_id']}")
                seen.add(row["frame_id"])
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara o dataset YOLO V10 de referência.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows([args.train, args.validation])
    # Agrupa pela imagem para suportar várias estruturas anotadas no mesmo frame.
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_path"]].append(row)
    manifest_rows: list[dict[str, object]] = []
    for frame_id, group in grouped.items():
        row = group[0]
        output_frame_id = row["frame_id"] if len(group) == 1 else f"{row['frame_id']}_multitarget"
        split = "train" if row["split"] == "train" else "val"
        source = Path(row["image_path"])
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(source)
        height, width = image.shape[:2]
        image_path = args.dataset_dir / "images" / split / f"{output_frame_id}.jpg"
        label_path = args.dataset_dir / "labels" / split / f"{output_frame_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_path)
        positives = [item for item in group if item["label"] == "goal_structure"]
        label_path.write_text(
            "\n".join(yolo_line(item, width, height) for item in positives)
            + ("\n" if positives else ""),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "frame_id": output_frame_id,
                "match_id": row["match_id"],
                "split": split,
                "label": row["label"],
                "image_path": str(image_path),
                "label_path": str(label_path),
                "structures": len(positives),
                "width": width,
                "height": height,
            }
        )

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    data_yaml = (
        f"path: {args.dataset_dir.resolve()}\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: [goal_structure]\n"
    )
    (args.dataset_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    summary = {
        "version": "V10_reference_dataset_v1",
        "train_frames": sum(row["split"] == "train" for row in manifest_rows),
        "validation_frames": sum(row["split"] == "val" for row in manifest_rows),
        "train_positive_frames": sum(row["split"] == "train" and row["structures"] > 0 for row in manifest_rows),
        "train_negative_frames": sum(row["split"] == "train" and row["structures"] == 0 for row in manifest_rows),
        "validation_positive_frames": sum(row["split"] == "val" and row["structures"] > 0 for row in manifest_rows),
        "validation_negative_frames": sum(row["split"] == "val" and row["structures"] == 0 for row in manifest_rows),
        "status": "ready_for_reference_training",
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
