"""Prepara comparações lado a lado de auditorias de dois detectores de bola."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def title(image, text: str, color: tuple[int, int, int]):
    """Acrescenta uma faixa identificando a versão exibida."""
    band = image.copy()
    band[:] = (20, 20, 20)
    cv2.putText(band, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    return cv2.vconcat([band[:55], image])


def main() -> int:
    parser = argparse.ArgumentParser(description="Cria comparação visual de duas auditorias da bola.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-name", default="YOLO11n")
    parser.add_argument("--candidate-name", default="YOLO11s")
    args = parser.parse_args()

    baseline = {row["sample_id"]: row for row in read_rows(args.baseline)}
    candidate = {row["sample_id"]: row for row in read_rows(args.candidate)}
    changed = [
        sample_id
        for sample_id in baseline
        if sample_id in candidate and baseline[sample_id]["preliminary_status"] != candidate[sample_id]["preliminary_status"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for index, sample_id in enumerate(changed, start=1):
        base, trial = baseline[sample_id], candidate[sample_id]
        base_image = cv2.imread(base["overlay_path"])
        trial_image = cv2.imread(trial["overlay_path"])
        if base_image is None or trial_image is None:
            raise FileNotFoundError(sample_id)
        if base_image.shape != trial_image.shape:
            trial_image = cv2.resize(trial_image, (base_image.shape[1], base_image.shape[0]))
        comparison = cv2.hconcat([
            title(base_image, f"{args.baseline_name}: {base['preliminary_status']}", (0, 220, 255)),
            title(trial_image, f"{args.candidate_name}: {trial['preliminary_status']}", (0, 255, 0)),
        ])
        overlay_path = args.output / "overlays" / f"{index:03d}_{sample_id}.jpg"
        overlay_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(overlay_path), comparison)
        manifest.append({
            "sample_id": sample_id,
            "match_id": base["match_id"],
            "timestamp_hhmmss": base["timestamp_hhmmss"],
            "ball_visibility": base["ball_visibility"],
            "baseline_status": base["preliminary_status"],
            "candidate_status": trial["preliminary_status"],
            "overlay_path": overlay_path.as_posix(),
            "review_status": "pending",
            "review_label": "",
            "review_notes": "",
        })
    target = args.output / "comparison_manifest.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"comparacoes={len(manifest)}")
    print(f"manifest={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
