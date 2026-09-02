"""Avalia uma regra conservadora de possível gol em janelas de eventos anotados."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def center(box_json: str) -> tuple[float, float] | None:
    if not box_json:
        return None
    x1, y1, x2, y2 = (float(value) for value in json.loads(box_json))
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def inside(points_json: str, point: tuple[float, float] | None) -> bool:
    if not points_json or point is None:
        return False
    return cv2.pointPolygonTest(np.array(json.loads(points_json), dtype="float32"), point, False) >= 0


def polygon_center(points_json: str) -> tuple[float, float] | None:
    if not points_json:
        return None
    points = np.array(json.loads(points_json), dtype="float32")
    return (float(points[:, 0].mean()), float(points[:, 1].mean()))


def distance(point: tuple[float, float] | None, reference: tuple[float, float] | None) -> float | None:
    if point is None or reference is None:
        return None
    return math.hypot(point[0] - reference[0], point[1] - reference[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia regra conservadora de possível gol em janelas anotadas.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--tracking-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-matches", nargs="+", default=None)
    parser.add_argument("--prior-frames", type=int, default=5)
    parser.add_argument("--minimum-approach-pixels", type=float, default=15.0)
    args = parser.parse_args()
    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequence_rows = list(csv.DictReader(file))
    with args.tracking_results.open(newline="", encoding="utf-8") as file:
        tracking = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    if args.include_matches:
        matches = set(args.include_matches)
        sequence_rows = [row for row in sequence_rows if row["match_id"] in matches]
        if not sequence_rows:
            parser.error("Nenhuma sequência corresponde às partidas informadas.")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sequence_rows:
        grouped[row["sequence_id"]].append(row)
    output_rows: list[dict[str, str]] = []
    for sequence_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["sequence_frame"]))
        event_index = next((index for index, row in enumerate(rows) if row.get("is_event_frame") == "True"), None)
        if event_index is None:
            continue
        event_row = rows[event_index]
        region = event_row.get("goal_region_points_json", "")
        volume = event_row.get("goal_volume_points_json", "")
        reference = polygon_center(region) or polygon_center(volume)
        event_result = tracking[(sequence_id, event_row["sequence_frame"])]
        event_center = center(event_result.get("selected_box_json", ""))
        in_goal = inside(region, event_center) or inside(volume, event_center)
        start_index = max(0, event_index - args.prior_frames)
        local_rows = rows[start_index:event_index + 1]
        local_centers = [center(tracking[(sequence_id, row["sequence_frame"])].get("selected_box_json", "")) for row in local_rows]
        detected_count = sum(point is not None for point in local_centers)
        previous = next((point for point in reversed(local_centers[:-1]) if point is not None), None)
        previous_distance, event_distance = distance(previous, reference), distance(event_center, reference)
        approach = (
            previous_distance is not None and event_distance is not None
            and previous_distance - event_distance >= args.minimum_approach_pixels
        )
        decision = "POSSIVEL_GOL_CANDIDATO" if in_goal and detected_count >= 2 and approach else "NOT_EVALUABLE"
        output_rows.append({
            "event_id": event_row["event_id"], "match_id": event_row["match_id"], "sequence_id": sequence_id,
            "selected_at_event": str(event_center is not None), "inside_goal_region": str(in_goal),
            "local_selected_frames": str(detected_count), "previous_distance_pixels": f"{previous_distance:.3f}" if previous_distance is not None else "",
            "event_distance_pixels": f"{event_distance:.3f}" if event_distance is not None else "",
            "approach_evidence": str(approach), "decision": decision,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "conservative_event_rule.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "experiment_id": args.output.name, "events": len(output_rows),
        "possible_goal_candidates": sum(row["decision"] == "POSSIVEL_GOL_CANDIDATO" for row in output_rows),
        "not_evaluable": sum(row["decision"] == "NOT_EVALUABLE" for row in output_rows),
        "note": "Simulação em janela centralizada no instante anotado; não é detector de eventos em vídeo completo.",
    }
    (args.output / "conservative_event_rule_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
