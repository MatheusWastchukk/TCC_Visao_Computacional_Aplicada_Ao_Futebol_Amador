"""Cria um dataset YOLO com sobreamostragem de exemplos difíceis."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def copy_split(source: Path, target: Path, split: str) -> int:
    count = 0
    for kind in ("images", "labels"):
        for file in (source / kind / split).rglob("*"):
            if file.is_file():
                destination = target / kind / split / file.relative_to(source / kind / split)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, destination)
                count += 1 if kind == "images" else 0
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara dataset com exemplos difíceis sobreamostrados.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("repeat deve ser maior ou igual a 1")
    train_count = copy_split(args.base, args.output, "train")
    validation_count = copy_split(args.base, args.output, "validation")
    with args.hard.open(newline="", encoding="utf-8") as file:
        hard_rows = list(csv.DictReader(file))
    added = 0
    for row in hard_rows:
        sample_id = row["sample_id"]
        image_source = args.base / "images" / "train" / f"{sample_id}.jpg"
        label_source = args.base / "labels" / "train" / f"{sample_id}.txt"
        if not image_source.exists() or not label_source.exists():
            raise FileNotFoundError(f"Amostra não encontrada no dataset base: {sample_id}")
        for copy_index in range(1, args.repeat + 1):
            suffix = f"_hard{copy_index}"
            shutil.copy2(image_source, args.output / "images" / "train" / f"{sample_id}{suffix}.jpg")
            shutil.copy2(label_source, args.output / "labels" / "train" / f"{sample_id}{suffix}.txt")
            added += 1
    (args.output / "SUMMARY.md").write_text(
        "# Dataset BALL-DATASET-V3-HARD\n\n"
        f"Imagens base de treino: {train_count}.\n"
        f"Imagens base de validação: {validation_count}.\n"
        f"Exemplos difíceis: {len(hard_rows)}.\n"
        f"Imagens difíceis adicionadas: {added}.\n"
        "A validação foi copiada sem alteração.\n",
        encoding="utf-8",
    )
    print(f"train_base={train_count}")
    print(f"validation={validation_count}")
    print(f"hard={len(hard_rows)}")
    print(f"hard_adicionados={added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
