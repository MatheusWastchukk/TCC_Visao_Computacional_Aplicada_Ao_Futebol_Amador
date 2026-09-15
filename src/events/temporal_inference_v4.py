"""Constrói e pontua o pool temporal operacional sem timestamps de referência."""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def temporal_groups(frame_rows: Iterable[dict[str, str]], merge_gap_seconds: float = 3.0, minimum_hits: int = 2) -> list[dict[str, object]]:
    """Reconstrói os mesmos atributos temporais V1 a partir dos hits espaciais."""
    by_match: dict[str, list[dict[str, float]]] = defaultdict(list)
    filenames: dict[str, str] = {}
    for row in frame_rows:
        box = json.loads(row["ball_box_json"])
        by_match[row["match_id"]].append({
            "time": float(row["timestamp_seconds"]), "score": float(row["candidate_score"]),
            "ball": float(row["ball_confidence"]), "goal": float(row["goal_confidence"]),
            "distance": float(row["normalized_distance"]), "inside": float(row["ball_inside_expanded_goal"] == "True"),
            "x": (float(box[0]) + float(box[2])) / 2.0, "y": (float(box[1]) + float(box[3])) / 2.0,
        })
        filenames[row["match_id"]] = row["video_filename"]
    output: list[dict[str, object]] = []
    for match_id, hits in by_match.items():
        grouped: list[list[dict[str, float]]] = []
        for hit in sorted(hits, key=lambda item: item["time"]):
            if not grouped or hit["time"] - grouped[-1][-1]["time"] > merge_gap_seconds:
                grouped.append([hit])
            else:
                grouped[-1].append(hit)
        for index, group in enumerate((item for item in grouped if len(item) >= minimum_hits), start=1):
            times = np.array([item["time"] for item in group])
            distance = np.array([item["distance"] for item in group])
            score = np.array([item["score"] for item in group])
            ball = np.array([item["ball"] for item in group])
            goal = np.array([item["goal"] for item in group])
            minimum = int(np.argmin(distance))
            gaps = np.diff(np.unique(times))
            speeds = [np.hypot(current["x"] - previous["x"], current["y"] - previous["y"]) / (current["time"] - previous["time"])
                      for previous, current in zip(group, group[1:]) if current["time"] > previous["time"]]
            values: dict[str, float] = {
                "hit_count": float(len(group)), "unique_timestamps": float(len(np.unique(times))),
                "duration_seconds": float(times[-1] - times[0]), "max_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
                "gap_count": float(len(gaps)), "max_candidate_score": float(score.max()), "mean_candidate_score": float(score.mean()),
                "max_ball_confidence": float(ball.max()), "mean_ball_confidence": float(ball.mean()),
                "max_goal_confidence": float(goal.max()), "mean_goal_confidence": float(goal.mean()),
                "min_normalized_distance": float(distance.min()), "mean_normalized_distance": float(distance.mean()), "distance_std": float(distance.std()),
                "distance_before_min": float(distance[:minimum].mean()) if minimum else float(distance[minimum]),
                "distance_after_min": float(distance[minimum + 1:].mean()) if minimum + 1 < len(distance) else float(distance[minimum]),
                "distance_slope": float(np.polyfit(times - times[0], distance, 1)[0]) if len(np.unique(times)) > 1 else 0.0,
                "inside_goal_fraction": float(np.mean([item["inside"] for item in group])),
                "mean_relative_speed": float(np.mean(speeds)) if speeds else 0.0, "max_relative_speed": float(np.max(speeds)) if speeds else 0.0,
            }
            duration = float(times[-1] - times[0])
            relative = (times - times[0]) / duration if duration > 0 else np.zeros_like(times)
            for bin_index in range(4):
                lower, upper = bin_index / 4, (bin_index + 1) / 4
                mask = (relative >= lower) & (relative < upper if bin_index < 3 else relative <= upper)
                values[f"bin_{bin_index}_mean_distance"] = float(distance[mask].mean()) if mask.any() else 0.0
                values[f"bin_{bin_index}_mean_score"] = float(score[mask].mean()) if mask.any() else 0.0
                values[f"bin_{bin_index}_mean_ball_confidence"] = float(ball[mask].mean()) if mask.any() else 0.0
                values[f"bin_{bin_index}_inside_fraction"] = float(np.mean([group[item]["inside"] for item in np.where(mask)[0]])) if mask.any() else 0.0
            output.append({"group_uid": f"{match_id}::temporal_group_{index:04d}", "match_id": match_id, "video_filename": filenames[match_id], "best_timestamp_seconds": float(times[int(np.argmax(score))]), **values})
    return output


def select_wide_pool(groups: Iterable[dict[str, object]], ranker: object, limit: int = 60, separation_seconds: float = 45.0) -> list[dict[str, str]]:
    """Aplica o Random Forest congelado e retorna a lista diversa top 60 em 20+40."""
    frozen = ranker
    features = frozen["features"]
    rows = list(groups)
    matrix = np.array([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
    scores = frozen["model"].predict_proba(matrix)[:, 1]
    ranked = [{**row, "rank_score": float(score)} for row, score in zip(rows, scores)]
    selected: list[dict[str, object]] = []
    for row in sorted(ranked, key=lambda item: float(item["rank_score"]), reverse=True):
        if any(abs(float(row["best_timestamp_seconds"]) - float(other["best_timestamp_seconds"])) < separation_seconds for other in selected):
            continue
        selected.append(row)
        if len(selected) == limit:
            break
    return [{
        "event_id": f"{row['match_id']}_wide_pool_{index:03d}", "match_id": str(row["match_id"]), "video_filename": str(row["video_filename"]),
        "event_timestamp_seconds": f"{float(row['best_timestamp_seconds']):.6f}", "rank_score": f"{float(row['rank_score']):.8f}",
        "candidate_start_seconds": f"{max(0.0, float(row['best_timestamp_seconds']) - 10.0):.6f}", "candidate_end_seconds": f"{float(row['best_timestamp_seconds']) + 20.0:.6f}",
        "delivery_start_seconds": f"{max(0.0, float(row['best_timestamp_seconds']) - 20.0):.6f}", "delivery_end_seconds": f"{float(row['best_timestamp_seconds']) + 40.0:.6f}",
        "temporal_rank": str(index), "source_interval_id": str(row["group_uid"]),
    } for index, row in enumerate(selected, start=1)]
