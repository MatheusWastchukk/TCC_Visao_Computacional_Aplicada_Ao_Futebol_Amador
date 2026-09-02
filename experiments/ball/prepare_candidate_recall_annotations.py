"""Prepara caixas de referência para medir candidate recall em casos difíceis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from prepare_ball_dataset_candidates import FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara a anotação diagnóstica da bola da jogada ausente entre candidatas."
    )
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.identity_manifest.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    selected = [row for row in source_rows if row.get("target_visibility") == "NAO_DETECTADA"]
    if not selected:
        raise ValueError("Não há casos NAO_DETECTADA no manifesto de identidade.")
    output_rows: list[dict[str, str]] = []
    for row in selected:
        output_rows.append({
            "sample_id": f"candidate_recall_{row['ambiguous_id']}",
            "match_id": row["match_id"],
            "video_filename": row["video_filename"],
            "capture_domain": "diagnostico_validation",
            "split": "diagnostic",
            "timestamp_seconds": "",
            "timestamp_hhmmss": row["timestamp_hhmmss"],
            "image_path": row["frame_path"],
            "ball_visibility": "pending",
            "ball_x1": "",
            "ball_y1": "",
            "ball_x2": "",
            "ball_y2": "",
            "annotation_status": "pending",
            "annotator": "",
            "notes": (
                f"origem=BALL-IDENTITY-EXP-0001; sequence_id={row['sequence_id']}; "
                "uso=diagnostico_candidate_recall; nao_usar_em_treino"
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"frames_diagnosticos={len(output_rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
