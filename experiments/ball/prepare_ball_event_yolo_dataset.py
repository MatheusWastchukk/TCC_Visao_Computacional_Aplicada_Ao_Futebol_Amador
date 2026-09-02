"""Materializa anotações de lances difíceis em dataset YOLO separado por partida."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2


def write_label(path: Path, row: dict[str, str]) -> None:
    """Escreve uma caixa YOLO para bola visível/parcial ou rótulo vazio quando não visível."""
    lines: list[str] = []
    if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in json.loads(row["ball_of_play_box_json"]))
        lines.append(f"0 {((x1 + x2) / 2) / width:.8f} {((y1 + y2) / 2) / height:.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara dataset YOLO de lances de gol difíceis por partida.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-matches", nargs="+", required=True)
    args = parser.parse_args()
    validation_matches = set(args.validation_matches)
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    included = [row for row in rows if row["ball_visibility"] in {"VISIBLE", "PARTIAL", "NOT_VISIBLE"}]
    if not included:
        raise ValueError("Não há anotações elegíveis.")
    manifest: list[dict[str, str]] = []
    for row in included:
        split = "validation" if row["match_id"] in validation_matches else "train"
        source = Path(row["image_path"])
        image_target = args.output / "images" / split / f"{row['sample_id']}.jpg"
        label_target = args.output / "labels" / split / f"{row['sample_id']}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_target)
        write_label(label_target, row)
        manifest.append({
            "sample_id": row["sample_id"], "event_id": row["event_id"], "match_id": row["match_id"], "split": split,
            "ball_visibility": row["ball_visibility"], "image_path": image_target.as_posix(), "label_path": label_target.as_posix(),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (args.output / "data.yaml").write_text(
        f"path: {args.output.as_posix()}\ntrain: images/train\nval: images/validation\nnames:\n  0: ball\n",
        encoding="utf-8",
    )
    counts = {split: sum(row["split"] == split for row in manifest) for split in ("train", "validation")}
    visible = {split: sum(row["split"] == split and row["ball_visibility"] in {"VISIBLE", "PARTIAL"} for row in manifest) for split in counts}
    print(json.dumps({"included": len(manifest), "counts": counts, "visible_or_partial": visible, "excluded_uncertain": len(rows) - len(included)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
