"""Extrai evidências de trajetória local de saídas já mineradas.

O experimento não modifica o ranking V4. Ele apenas transforma
``candidate_frames_v1.csv`` em atributos por grupo temporal para que a
hipótese seja avaliada em desenvolvimento antes de qualquer confirmação.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.events.local_trajectory_v1 import best_local_trajectory, observations_from_frame_rows


FIELDS = [
    "group_uid", "match_id", "video_filename", "hit_count", "unique_timestamps",
    "path_observations", "path_score", "mean_normalized_step", "continuity",
    "first_timestamp_seconds", "last_timestamp_seconds",
]


def temporal_groups(rows: list[dict[str, str]], merge_gap_seconds: float, minimum_hits: int) -> list[list[dict[str, str]]]:
    """Agrupa hits com a mesma regra temporal do minerador V4."""
    grouped_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_by_match[row["match_id"]].append(row)
    groups: list[list[dict[str, str]]] = []
    for match_rows in grouped_by_match.values():
        current: list[dict[str, str]] = []
        for row in sorted(match_rows, key=lambda item: float(item["timestamp_seconds"])):
            if current and float(row["timestamp_seconds"]) - float(current[-1]["timestamp_seconds"]) > merge_gap_seconds:
                if len(current) >= minimum_hits:
                    groups.append(current)
                current = []
            current.append(row)
        if len(current) >= minimum_hits:
            groups.append(current)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai trajetória local de grupos temporais já minerados.")
    parser.add_argument("--frames", type=Path, nargs="+", required=True, help="Um ou mais candidate_frames_v1.csv já existentes.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-gap-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-hits", type=int, default=2)
    parser.add_argument("--max-path-gap-seconds", type=float, default=1.0)
    parser.add_argument("--transition-penalty", type=float, default=2.0)
    args = parser.parse_args()
    if args.merge_gap_seconds <= 0 or args.minimum_hits < 2 or args.max_path_gap_seconds <= 0 or args.transition_penalty < 0:
        parser.error("Os parâmetros de agrupamento e trajetória devem ser positivos; minimum-hits deve ser ao menos 2.")
    frame_rows: list[dict[str, str]] = []
    for frames_path in args.frames:
        with frames_path.open(newline="", encoding="utf-8") as file:
            frame_rows.extend(csv.DictReader(file))
    required = {"match_id", "video_filename", "timestamp_seconds", "ball_box_json", "goal_box_json", "ball_confidence"}
    if not frame_rows or required - set(frame_rows[0]):
        raise ValueError("O CSV não possui as colunas de hits requeridas pelo experimento.")
    output_rows: list[dict[str, str]] = []
    group_indexes: dict[str, int] = defaultdict(int)
    for group in temporal_groups(frame_rows, args.merge_gap_seconds, args.minimum_hits):
        trajectory = best_local_trajectory(
            observations_from_frame_rows(group),
            max_gap_seconds=args.max_path_gap_seconds,
            transition_penalty=args.transition_penalty,
        )
        first = group[0]
        group_indexes[first["match_id"]] += 1
        times = [float(row["timestamp_seconds"]) for row in group]
        output_rows.append({
            "group_uid": f"{first['match_id']}::temporal_group_{group_indexes[first['match_id']]:04d}",
            "match_id": first["match_id"], "video_filename": first["video_filename"],
            "hit_count": str(len(group)), "unique_timestamps": str(len(set(times))),
            "path_observations": str(len(trajectory.observations)),
            "path_score": f"{trajectory.path_score:.8f}",
            "mean_normalized_step": f"{trajectory.mean_normalized_step:.8f}",
            "continuity": f"{trajectory.continuity:.8f}",
            "first_timestamp_seconds": f"{min(times):.6f}", "last_timestamp_seconds": f"{max(times):.6f}",
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "local_trajectory_groups_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    metrics = {
        "experiment_id": args.output.name,
        "scope": "atributos diagnósticos; não altera o V4",
        "groups": len(output_rows),
        "groups_by_match": dict(sorted(group_indexes.items())),
        "source_files": [str(path) for path in args.frames],
        "mean_continuity": sum(float(row["continuity"]) for row in output_rows) / len(output_rows) if output_rows else 0.0,
        "parameters": {
            "merge_gap_seconds": args.merge_gap_seconds,
            "minimum_hits": args.minimum_hits,
            "max_path_gap_seconds": args.max_path_gap_seconds,
            "transition_penalty": args.transition_penalty,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
