"""Reconstrói grupos TRAIN e extrai atributos temporais para ranking de eventos."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FEATURES = [
    "hit_count", "unique_timestamps", "duration_seconds", "max_gap_seconds", "gap_count",
    "max_candidate_score", "mean_candidate_score", "max_ball_confidence", "mean_ball_confidence",
    "max_goal_confidence", "mean_goal_confidence", "min_normalized_distance",
    "mean_normalized_distance", "distance_std", "distance_before_min", "distance_after_min",
    "distance_slope", "inside_goal_fraction", "mean_relative_speed", "max_relative_speed",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara dataset temporal de grupos, somente TRAIN.")
    parser.add_argument("--hit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-gap-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-hits", type=int, default=2)
    parser.add_argument("--positive-radius-seconds", type=float, default=10.0)
    parser.add_argument("--temporal-bins", type=int, default=0)
    args = parser.parse_args()
    if args.temporal_bins < 0:
        raise ValueError("temporal-bins deve ser não negativo.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}.")

    hits_by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in args.hit_files:
        rows, fields = read_csv(path)
        required = {"match_id", "timestamp_seconds", "candidate_score", "ball_confidence", "goal_confidence", "normalized_distance", "ball_inside_expanded_goal", "ball_box_json"}
        if missing := required - set(fields):
            raise ValueError(f"Hits sem colunas em {path}: {sorted(missing)}")
        for row in rows:
            box = json.loads(row["ball_box_json"])
            hits_by_match[row["match_id"]].append({
                "timestamp": float(row["timestamp_seconds"]), "score": float(row["candidate_score"]),
                "ball_confidence": float(row["ball_confidence"]), "goal_confidence": float(row["goal_confidence"]),
                "distance": float(row["normalized_distance"]), "inside": row["ball_inside_expanded_goal"] == "True",
                "center_x": (float(box[0]) + float(box[2])) / 2.0,
                "center_y": (float(box[1]) + float(box[3])) / 2.0,
            })
    goals, fields = read_csv(args.goal_timestamps)
    if any(row["split"] != "train" for row in goals):
        raise AssertionError("O dataset temporal aceita somente TRAIN; TEST permanece fechado.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in goals:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))

    binned_features = [
        f"bin_{index}_{name}"
        for index in range(args.temporal_bins)
        for name in ("mean_distance", "mean_score", "mean_ball_confidence", "inside_fraction")
    ]
    feature_names = FEATURES + binned_features
    dataset: list[dict[str, object]] = []
    for match_id, hits in hits_by_match.items():
        groups: list[list[dict[str, object]]] = []
        for hit in sorted(hits, key=lambda item: float(item["timestamp"])):
            if not groups or float(hit["timestamp"]) - float(groups[-1][-1]["timestamp"]) > args.merge_gap_seconds:
                groups.append([hit])
            else:
                groups[-1].append(hit)
        for index, group in enumerate((group for group in groups if len(group) >= args.minimum_hits), start=1):
            times = np.array([float(hit["timestamp"]) for hit in group])
            distances = np.array([float(hit["distance"]) for hit in group])
            scores = np.array([float(hit["score"]) for hit in group])
            balls = np.array([float(hit["ball_confidence"]) for hit in group])
            goal_confidences = np.array([float(hit["goal_confidence"]) for hit in group])
            min_index = int(np.argmin(distances))
            gaps = np.diff(np.unique(times))
            speeds = []
            for previous, current in zip(group, group[1:]):
                delta_time = float(current["timestamp"]) - float(previous["timestamp"])
                if delta_time > 0:
                    speeds.append(np.hypot(float(current["center_x"]) - float(previous["center_x"]), float(current["center_y"]) - float(previous["center_y"])) / delta_time)
            best_time = float(times[int(np.argmax(scores))])
            values = {
                "hit_count": len(group), "unique_timestamps": len(np.unique(times)),
                "duration_seconds": float(times[-1] - times[0]), "max_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
                "gap_count": int(len(gaps)), "max_candidate_score": float(scores.max()), "mean_candidate_score": float(scores.mean()),
                "max_ball_confidence": float(balls.max()), "mean_ball_confidence": float(balls.mean()),
                "max_goal_confidence": float(goal_confidences.max()), "mean_goal_confidence": float(goal_confidences.mean()),
                "min_normalized_distance": float(distances.min()), "mean_normalized_distance": float(distances.mean()),
                "distance_std": float(distances.std()),
                "distance_before_min": float(distances[:min_index].mean()) if min_index else float(distances[min_index]),
                "distance_after_min": float(distances[min_index + 1:].mean()) if min_index + 1 < len(distances) else float(distances[min_index]),
                "distance_slope": float(np.polyfit(times - times[0], distances, 1)[0]) if len(np.unique(times)) > 1 else 0.0,
                "inside_goal_fraction": float(np.mean([bool(hit["inside"]) for hit in group])),
                "mean_relative_speed": float(np.mean(speeds)) if speeds else 0.0,
                "max_relative_speed": float(np.max(speeds)) if speeds else 0.0,
            }
            if args.temporal_bins:
                duration = float(times[-1] - times[0])
                relative = (times - times[0]) / duration if duration > 0 else np.zeros_like(times)
                for bin_index in range(args.temporal_bins):
                    lower = bin_index / args.temporal_bins
                    upper = (bin_index + 1) / args.temporal_bins
                    mask = (relative >= lower) & (relative < upper if bin_index + 1 < args.temporal_bins else relative <= upper)
                    values[f"bin_{bin_index}_mean_distance"] = float(distances[mask].mean()) if mask.any() else 0.0
                    values[f"bin_{bin_index}_mean_score"] = float(scores[mask].mean()) if mask.any() else 0.0
                    values[f"bin_{bin_index}_mean_ball_confidence"] = float(balls[mask].mean()) if mask.any() else 0.0
                    values[f"bin_{bin_index}_inside_fraction"] = float(np.mean([bool(group[item]["inside"]) for item in np.where(mask)[0]])) if mask.any() else 0.0
            dataset.append({
                "group_uid": f"{match_id}::temporal_group_{index:04d}", "match_id": match_id,
                "best_timestamp_seconds": f"{best_time:.6f}",
                "weak_label": int(any(abs(best_time - goal) <= args.positive_radius_seconds for goal in goals_by_match[match_id])),
                **{key: f"{value:.6f}" for key, value in values.items()},
            })
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "temporal_group_dataset_v1.csv", ["group_uid", "match_id", "best_timestamp_seconds", "weak_label", *feature_names], dataset)
    summary = {"experiment_id": "GOAL-EVENT-TEMPORAL-DATASET-V1", "scope": "somente TRAIN", "groups": len(dataset), "positive_groups": sum(int(row["weak_label"]) for row in dataset), "features": feature_names, "positive_radius_seconds": args.positive_radius_seconds, "temporal_bins": args.temporal_bins}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
