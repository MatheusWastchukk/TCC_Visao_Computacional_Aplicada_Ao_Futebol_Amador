"""Cria linhas adicionais para anotar múltiplas estruturas na mesma imagem."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TARGET_FRAMES = {
    "match_12_validation_coverage_03",
    "match_14_validation_coverage_04",
    "match_14_validation_coverage_07",
    "match_14_validation_coverage_08",
    "match_26_validation_coverage_03",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara candidatos multi-target da validation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-targets", type=int, default=2)
    parser.add_argument("--start-target", type=int, default=2)
    parser.add_argument(
        "--frame-ids",
        help="Lista opcional de frames separados por vírgula; por padrão usa os cinco frames conhecidos.",
    )
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    target_frames = (
        {item.strip() for item in args.frame_ids.split(",") if item.strip()}
        if args.frame_ids
        else TARGET_FRAMES
    )
    selected = [row for row in source_rows if row["frame_id"] in target_frames]
    if len(selected) != len(target_frames):
        found = {row["frame_id"] for row in selected}
        raise ValueError(f"Frames ausentes: {sorted(target_frames - found)}")
    rows: list[dict[str, str]] = []
    for source in selected:
        for target_index in range(args.start_target, args.start_target + args.additional_targets):
            row = {field: "" for field in source}
            row.update(
                {
                    "frame_id": f"{source['frame_id']}__structure_{target_index:02d}",
                    "match_id": source["match_id"],
                    "video_filename": source["video_filename"],
                    "timestamp_seconds": source["timestamp_seconds"],
                    "original_label": "pending",
                    "label": "pending",
                    "split": "validation",
                    "image_path": source["image_path"],
                    "annotation_status": "pending",
                    "notes": f"multi-target; anotar estrutura adicional {target_index:02d}; se não existir, usar N",
                }
            )
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"imagens={len(selected)}")
    print(f"linhas_criadas={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
