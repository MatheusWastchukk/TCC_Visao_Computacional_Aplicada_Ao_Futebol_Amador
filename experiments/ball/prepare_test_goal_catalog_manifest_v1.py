"""Cria o catálogo de timestamp TEST pendente após o freeze V2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v2.json"
SPLIT_MANIFEST = ROOT / "experiments/datasets/dataset_split_v1.csv"
VIDEOS = ROOT / "data/Videos_Test_Fechado"
DEFAULT_OUT = ROOT / "experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V1/goal_catalog_timestamp_manifest_v1.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not FREEZE.is_file():
        raise RuntimeError("O freeze V2 precisa existir antes de criar a fila TEST.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("manifest_version") != 2 or freeze.get("is_final_sprint_freeze") is not True:
        raise RuntimeError("O manifesto V2 não confirma o freeze final.")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"A fila TEST já existe; não será sobrescrita: {output}")

    with SPLIT_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        matches = [row for row in csv.DictReader(stream) if row.get("split") == "test"]
    if len(matches) != 9:
        raise ValueError(f"Esperadas 9 partidas TEST; encontradas {len(matches)}.")
    fields = ["goal_catalog_id", "match_id", "video_filename", "split", "catalog_goal_index", "event_timestamp_seconds", "event_timestamp_hhmmss", "annotation_status", "notes"]
    rows = []
    for match in matches:
        video = VIDEOS / match["video_filename"]
        if not video.is_file():
            raise FileNotFoundError(f"Vídeo indicado no split manifest não encontrado: {video}")
        for index in range(1, int(match["goals_registered"]) + 1):
            rows.append({
                "goal_catalog_id": f"{match['match_id']}__catalog_goal_{index:03d}",
                "match_id": match["match_id"],
                "video_filename": match["video_filename"],
                "split": "test",
                "catalog_goal_index": str(index),
                "event_timestamp_seconds": "",
                "event_timestamp_hhmmss": "",
                "annotation_status": "pending",
                "notes": "Timestamp a ser revisado no vídeo original após o freeze V2; ainda não avaliado pelo pipeline.",
            })
    if len(rows) != 57:
        raise ValueError(f"Esperados 57 gols registrados no split TEST; encontrados {len(rows)}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "TEST_ANNOTATION_TEMPLATE_CREATED", "matches": len(matches), "catalog_goals_pending": len(rows), "path": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
