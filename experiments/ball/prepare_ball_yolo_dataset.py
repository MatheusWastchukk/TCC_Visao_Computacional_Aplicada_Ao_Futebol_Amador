"""Converte as anotações da bola para o formato YOLO e preserva a visibilidade."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara o dataset YOLO da bola.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    included = [row for row in rows if row["ball_visibility"] in {"VISIBLE", "PARTIAL", "NOT_VISIBLE"}]
    manifest_rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in included:
        split = row["split"]
        image_source = Path(row["image_path"])
        image_target = args.output / "images" / split / f"{row['sample_id']}.jpg"
        label_target = args.output / "labels" / split / f"{row['sample_id']}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_target)
        lines: list[str] = []
        if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            image = cv2.imread(str(image_source))
            if image is None:
                raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_source}")
            image_height, image_width = image.shape[:2]
            x1, y1, x2, y2 = [float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")]
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            lines.append(f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}")
        label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "match_id": row["match_id"],
                "split": split,
                "ball_visibility": row["ball_visibility"],
                "image_path": image_target.as_posix(),
                "label_path": label_target.as_posix(),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    excluded = [row for row in rows if row["ball_visibility"] == "UNCERTAIN"]
    summary = (
        "# BALL-DATASET-V1 — Resumo da conversão\n\n"
        f"Incluídos: {len(included)} frames.\n"
        f"Train: {counts.get('train', 0)}. Validation: {counts.get('validation', 0)}.\n"
        f"Excluídos por incerteza: {len(excluded)}.\n"
        "`NOT_VISIBLE` foi convertido em imagem com arquivo de rótulo vazio e não é penalizado como falso negativo.\n"
    )
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"incluidos={len(included)}")
    print(f"train={counts.get('train', 0)}")
    print(f"validation={counts.get('validation', 0)}")
    print(f"uncertain_excluidos={len(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
