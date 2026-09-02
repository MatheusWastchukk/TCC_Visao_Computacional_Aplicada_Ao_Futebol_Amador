"""Avalia uma regra de entrada na goleira sem supor direção de aproximação."""

from __future__ import annotations

import argparse
import csv
import json
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
    points = np.array(json.loads(points_json), dtype="float32")
    return cv2.pointPolygonTest(points, point, False) >= 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia entrada no gol com persistência local, sem heurística de direção."
    )
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--tracking-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-matches", nargs="+", default=None)
    parser.add_argument("--prior-frames", type=int, default=5)
    parser.add_argument("--minimum-local-selected", type=int, default=2)
    args = parser.parse_args()

    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequences = list(csv.DictReader(file))
    with args.tracking_results.open(newline="", encoding="utf-8") as file:
        tracking = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    if args.include_matches:
        selected_matches = set(args.include_matches)
        sequences = [row for row in sequences if row["match_id"] in selected_matches]
        if not sequences:
            parser.error("Nenhuma sequência corresponde às partidas informadas.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sequences:
        grouped[row["sequence_id"]].append(row)
    output_rows: list[dict[str, str]] = []
    for sequence_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["sequence_frame"]))
        event_index = next((index for index, row in enumerate(rows) if row.get("is_event_frame") == "True"), None)
        if event_index is None:
            continue
        event_row = rows[event_index]
        observable = event_row.get("ball_visibility") in {"VISIBLE", "PARTIAL"}
        event_result = tracking[(sequence_id, event_row["sequence_frame"])]
        event_center = center(event_result.get("selected_box_json", ""))
        inside_front = inside(event_row.get("goal_region_points_json", ""), event_center)
        inside_volume = inside(event_row.get("goal_volume_points_json", ""), event_center)
        local_rows = rows[max(0, event_index - args.prior_frames):event_index + 1]
        local_selected = sum(
            center(tracking[(sequence_id, row["sequence_frame"])].get("selected_box_json", "")) is not None
            for row in local_rows
        )
        candidate = observable and (inside_front or inside_volume) and local_selected >= args.minimum_local_selected
        decision = "POSSIVEL_GOL_CANDIDATO" if candidate else "NOT_EVALUABLE"
        output_rows.append({
            "event_id": event_row["event_id"], "match_id": event_row["match_id"], "sequence_id": sequence_id,
            "ball_visibility": event_row.get("ball_visibility", ""), "selected_at_event": str(event_center is not None),
            "inside_front": str(inside_front), "inside_volume": str(inside_volume),
            "local_selected_frames": str(local_selected), "decision": decision,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "entry_persistence_event_rule.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "experiment_id": args.output.name,
        "events": len(output_rows),
        "possible_goal_candidates": sum(row["decision"] == "POSSIVEL_GOL_CANDIDATO" for row in output_rows),
        "not_evaluable": sum(row["decision"] == "NOT_EVALUABLE" for row in output_rows),
        "minimum_local_selected": args.minimum_local_selected,
        "note": "Simulação em janela centralizada no instante anotado; não é detector de eventos em vídeo completo.",
    }
    (args.output / "entry_persistence_event_rule_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
