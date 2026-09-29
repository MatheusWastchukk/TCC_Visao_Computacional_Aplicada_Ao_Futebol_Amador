"""Extrai evidência cinemática compatível com ação ofensiva da bola.

Os atributos não afirmam que houve chute. Eles serão usados somente em uma
comparação LOMO em TRAIN antes de qualquer confirmação externa.
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

from extract_local_trajectory_features_v1 import temporal_groups
from src.events.local_trajectory_v1 import best_local_trajectory, observations_from_frame_rows
from src.events.shot_motion_v1 import shot_motion_evidence


FIELDS = [
    "group_uid", "match_id", "video_filename", "shot_path_observations",
    "shot_mean_speed", "shot_max_speed", "shot_mean_toward_goal_cosine",
    "shot_approach_fraction", "shot_normalized_distance_delta",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai movimento da bola em direção ao gol de hits já minerados.")
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-gap-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-hits", type=int, default=2)
    parser.add_argument("--max-path-gap-seconds", type=float, default=1.0)
    parser.add_argument("--transition-penalty", type=float, default=2.0)
    args = parser.parse_args()
    if args.merge_gap_seconds <= 0 or args.minimum_hits < 2 or args.max_path_gap_seconds <= 0 or args.transition_penalty < 0:
        parser.error("Os parâmetros devem ser positivos; minimum-hits deve ser ao menos 2.")
    frame_rows: list[dict[str, str]] = []
    for frames_path in args.frames:
        with frames_path.open(newline="", encoding="utf-8") as file:
            frame_rows.extend(csv.DictReader(file))
    group_indexes: dict[str, int] = defaultdict(int)
    output_rows: list[dict[str, str]] = []
    for group in temporal_groups(frame_rows, args.merge_gap_seconds, args.minimum_hits):
        trajectory = best_local_trajectory(
            observations_from_frame_rows(group),
            max_gap_seconds=args.max_path_gap_seconds,
            transition_penalty=args.transition_penalty,
        )
        evidence = shot_motion_evidence(trajectory)
        first = group[0]
        group_indexes[first["match_id"]] += 1
        output_rows.append({
            "group_uid": f"{first['match_id']}::temporal_group_{group_indexes[first['match_id']]:04d}",
            "match_id": first["match_id"], "video_filename": first["video_filename"],
            "shot_path_observations": str(evidence.path_observations),
            "shot_mean_speed": f"{evidence.mean_speed:.8f}",
            "shot_max_speed": f"{evidence.max_speed:.8f}",
            "shot_mean_toward_goal_cosine": f"{evidence.mean_toward_goal_cosine:.8f}",
            "shot_approach_fraction": f"{evidence.approach_fraction:.8f}",
            "shot_normalized_distance_delta": f"{evidence.normalized_distance_delta:.8f}",
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "shot_motion_groups_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    metrics = {
        "experiment_id": args.output.name,
        "scope": "atributos cinemáticos diagnósticos; não detecta chute nem altera o V4",
        "groups": len(output_rows),
        "groups_by_match": dict(sorted(group_indexes.items())),
        "parameters": {"merge_gap_seconds": args.merge_gap_seconds, "minimum_hits": args.minimum_hits, "max_path_gap_seconds": args.max_path_gap_seconds, "transition_penalty": args.transition_penalty},
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
