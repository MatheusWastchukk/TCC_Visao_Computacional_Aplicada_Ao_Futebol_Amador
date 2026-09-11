"""Cria um subconjunto pequeno e reproduzível para validar o fluxo de treino."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria subset de smoke test do COCO RF-DETR.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--valid", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def build_split(source: Path, target: Path, count: int, rng: random.Random) -> int:
    payload = json.loads((source / "_annotations.coco.json").read_text(encoding="utf-8"))
    chosen = rng.sample(payload["images"], min(count, len(payload["images"])))
    chosen_ids = {image["id"] for image in chosen}
    annotations = [item for item in payload["annotations"] if item["image_id"] in chosen_ids]
    target_images = target / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    for image in chosen:
        relative = Path(image["file_name"])
        shutil.copy2(source / relative, target / relative)
    payload["images"] = chosen
    payload["annotations"] = annotations
    (target / "_annotations.coco.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(chosen)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    counts = {
        "train": build_split(args.source / "train", args.output / "train", args.train, rng),
        "valid": build_split(args.source / "valid", args.output / "valid", args.valid, rng),
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps({"seed": args.seed, **counts}, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
