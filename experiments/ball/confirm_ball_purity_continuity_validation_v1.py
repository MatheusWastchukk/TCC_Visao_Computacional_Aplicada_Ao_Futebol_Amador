"""Confirma em VALIDATION o ranker de trajetória com razões congeladas no TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "results/local_trajectory_dataset_train_v2/temporal_group_dataset_local_trajectory_v1.csv"
VALIDATION = ROOT / "results/local_trajectory_dataset_validation_v1/temporal_group_dataset_local_trajectory_v1.csv"
GOALS = ROOT / "results/local_trajectory_dataset_validation_v1/goal_timestamps_train_subset_v1.csv"
OUTPUT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PURITY-CONTINUITY-VALIDATION-V1"
EXCLUDED = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
RATIOS = ("detections_per_sampled_timestamp", "local_path_coverage", "local_path_share_of_hits", "ball_confidence_mean_to_max")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def add_ratios(row: dict[str, str]) -> dict[str, float]:
    hits = max(0.0, float(row["hit_count"]))
    samples = max(1.0, float(row["unique_timestamps"]))
    path_obs = max(0.0, float(row["local_trajectory_path_observations"]))
    return {
        RATIOS[0]: hits / samples,
        RATIOS[1]: path_obs / samples,
        RATIOS[2]: path_obs / max(hits, 1.0),
        RATIOS[3]: max(0.0, float(row["mean_ball_confidence"])) / max(float(row["max_ball_confidence"]), 1e-8),
    }


def select(rows: list[dict[str, object]], k: int, gap: float) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda x: (-float(x["rank_score"]), str(x["group_uid"]))):
        if all(abs(float(row["best_timestamp_seconds"]) - float(other["best_timestamp_seconds"])) >= gap for other in result):
            result.append(row)
            if len(result) == k:
                break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--goal-timestamps", type=Path, default=GOALS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limits", type=int, nargs="+", default=[20, 40, 60, 100])
    parser.add_argument("--gap-seconds", type=float, default=45.0)
    parser.add_argument("--pre-seconds", type=float, default=20.0)
    parser.add_argument("--post-seconds", type=float, default=40.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output already contains files: {args.output}")
    train, val = read_csv(args.train), read_csv(args.validation)
    goals: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(args.goal_timestamps):
        if row.get("split", "").lower() != "validation":
            raise AssertionError("Expected validation-only catalog; TEST must remain closed.")
        if row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds"):
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if len(train) != 5372 or len(val) != 1944 or sum(map(len, goals.values())) != 51:
        raise ValueError(f"Unexpected scope: train={len(train)}, validation={len(val)}, goals={sum(map(len, goals.values()))}")
    features = [c for c in train[0] if c not in EXCLUDED]
    if features != [c for c in val[0] if c not in EXCLUDED]:
        raise ValueError("TRAIN and VALIDATION feature columns/order differ.")
    train_enriched = [{**r, **add_ratios(r)} for r in train]
    val_enriched = [{**r, **add_ratios(r)} for r in val]
    y = np.asarray([int(r["weak_label"]) for r in train], dtype=np.int64)
    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    per_match: list[dict[str, object]] = []
    for method, added in (("local_trajectory_baseline", ()), ("all_four_frozen", RATIOS)):
        names = [*features, *added]
        x_train = np.asarray([[float(r[n]) for n in names] for r in train_enriched], dtype=np.float32)
        x_val = np.asarray([[float(r[n]) for n in names] for r in val_enriched], dtype=np.float32)
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(x_train, y)
        scores = model.predict_proba(x_val)[:, 1]
        by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r, score in zip(val, scores, strict=True):
            ranked = {**r, "rank_score": float(score)}
            by_match[r["match_id"]].append(ranked)
            predictions.append({"method": method, "match_id": r["match_id"], "group_uid": r["group_uid"],
                                "best_timestamp_seconds": r["best_timestamp_seconds"], "rank_score": f"{score:.8f}"})
        for k in args.limits:
            selected = {match: select(match_rows, k, args.gap_seconds) for match, match_rows in by_match.items()}
            covered = sum(any(max(0.0, float(c["best_timestamp_seconds"]) - args.pre_seconds) <= t <=
                              float(c["best_timestamp_seconds"]) + args.post_seconds
                              for c in selected[match]) for match, times in goals.items() for t in times)
            intervals = []
            per_match_covered = {}
            for match, chosen in selected.items():
                spans = sorted((max(0.0, float(c["best_timestamp_seconds"]) - args.pre_seconds),
                                float(c["best_timestamp_seconds"]) + args.post_seconds) for c in chosen)
                merged: list[list[float]] = []
                for start, end in spans:
                    if not merged or start > merged[-1][1]:
                        merged.append([start, end])
                    else:
                        merged[-1][1] = max(merged[-1][1], end)
                intervals.extend(merged)
                per_match_covered[match] = sum(any(a <= t <= b for a, b in merged) for t in goals[match])
                per_match.append({"method": method, "match_id": match, "catalog_goals": len(goals[match]),
                                  "covered_goals": per_match_covered[match], "k": k})
            minutes = sum(b - a for a, b in intervals) / 60 / len(by_match)
            results.append({"method": method, "k_per_match": k, "catalog_goals": 51, "covered_goals": covered,
                            "recall": f"{covered/51:.6f}", "selected_candidates": sum(map(len, selected.values())),
                            "unique_review_minutes_per_match": f"{minutes:.3f}"})
    args.output.mkdir(parents=True)
    write_csv(args.output / "recall_at_k_comparison_v1.csv", results)
    write_csv(args.output / "validation_predictions_v1.csv", predictions)
    write_csv(args.output / "per_match_recall_v1.csv", per_match)
    summary = {"experiment_id": args.output.name, "scope": "TRAIN full fit (5,372 groups); VALIDATION confirm (1,944 groups, 8 matches, 51 catalog goals); TEST closed",
               "features": features, "added_ratios": list(RATIOS),
               "models": "Paired RandomForestClassifier trained on all TRAIN weak labels: 300 trees, min_samples_leaf=2, class_weight=balanced, random_state=42",
               "selection": {"delivery_window_seconds": [args.pre_seconds, args.post_seconds], "gap_seconds": args.gap_seconds, "limits": args.limits},
               "warning": "One frozen VALIDATION confirmation of features specified from TRAIN/LOMO. VALIDATION results do not retune features or model. Weak TRAIN labels. No TEST access."}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary["scope"], "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
