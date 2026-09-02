"""Compara a evidência manual de entrada com a posição selecionada pelo tracker."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def center(box_json: str) -> tuple[float, float] | None:
    if not box_json:
        return None
    x1, y1, x2, y2 = (float(value) for value in json.loads(box_json))
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def contains(points_json: str, point: tuple[float, float] | None) -> bool | None:
    if not points_json or point is None:
        return None
    polygon = np.array(json.loads(points_json), dtype="float32")
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def evidence(front: bool | None, volume: bool | None, observable: bool) -> str:
    if not observable:
        return "NAO_OBSERVAVEL"
    if front:
        return "DENTRO_PLANO_ENTRADA"
    if volume:
        return "DENTRO_VOLUME_PROJETADO"
    return "FORA_DAS_REGIOES"


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia evidência espacial manual de entrada da bola em lances positivos.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--tracking-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-matches", nargs="+", default=None)
    args = parser.parse_args()
    with args.sequences.open(newline="", encoding="utf-8") as file:
        event_rows = [row for row in csv.DictReader(file) if row.get("is_event_frame") == "True"]
    with args.tracking_results.open(newline="", encoding="utf-8") as file:
        tracking = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    if args.include_matches:
        matches = set(args.include_matches)
        event_rows = [row for row in event_rows if row["match_id"] in matches]
        if not event_rows:
            parser.error("Nenhum evento corresponde às partidas informadas.")
    output_rows: list[dict[str, str]] = []
    for row in event_rows:
        tracker = tracking[(row["sequence_id"], row["sequence_frame"])]
        observable = row["ball_visibility"] in {"VISIBLE", "PARTIAL"}
        target_center = center(row["ball_of_play_box_json"]) if observable else None
        tracker_center = center(tracker.get("selected_box_json", ""))
        target_front = contains(row.get("goal_region_points_json", ""), target_center)
        target_volume = contains(row.get("goal_volume_points_json", ""), target_center)
        tracker_front = contains(row.get("goal_region_points_json", ""), tracker_center)
        tracker_volume = contains(row.get("goal_volume_points_json", ""), tracker_center)
        output_rows.append({
            "event_id": row["event_id"], "match_id": row["match_id"], "sequence_id": row["sequence_id"],
            "ball_visibility": row["ball_visibility"], "identity_correct": tracker["identity_correct"],
            "target_evidence": evidence(target_front, target_volume, observable),
            "tracker_evidence": evidence(tracker_front, tracker_volume, tracker_center is not None),
            "target_inside_front": str(target_front) if target_front is not None else "",
            "target_inside_volume": str(target_volume) if target_volume is not None else "",
            "tracker_inside_front": str(tracker_front) if tracker_front is not None else "",
            "tracker_inside_volume": str(tracker_volume) if tracker_volume is not None else "",
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "event_geometry_evidence.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "experiment_id": args.output.name,
        "events": len(output_rows),
        "target_evidence_counts": {label: sum(row["target_evidence"] == label for row in output_rows) for label in sorted({row["target_evidence"] for row in output_rows})},
        "tracker_evidence_counts": {label: sum(row["tracker_evidence"] == label for row in output_rows) for label in sorted({row["tracker_evidence"] for row in output_rows})},
    }
    (args.output / "event_geometry_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
