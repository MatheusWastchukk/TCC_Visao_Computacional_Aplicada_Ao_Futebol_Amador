"""Create paired per-match, bootstrap and Pareto summaries from nested OOF outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1"
DEFAULT_OUT = ROOT / "experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1"
BOOTSTRAP_SEED = 901006
BOOTSTRAP_REPLICATES = 10_000
MODE_SPECS = {
    "FAST_V2": (30, "NO_RESCUE"),
    "BALANCED_V2": (40, "CORE_RESCUE"),
    "HIGH_RECALL_V2": (58, "CORE_RESCUE"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(INPUT / "nested_time_budget_summary_v1.csv")
    outer_rows = read_csv(INPUT / "nested_time_budget_outer_folds_v1.csv")
    goal_rows = read_csv(INPUT / "nested_time_budget_goal_coverage_v1.csv")
    if len(outer_rows) != 25 * 5 * 3:
        raise ValueError(f"Expected 375 outer rows, found {len(outer_rows)}")
    if len(goal_rows) != 152 * 5 * 3:
        raise ValueError(f"Expected 2,280 goal rows, found {len(goal_rows)}")
    matches = sorted({row["outer_match"] for row in outer_rows})
    if len(matches) != 25:
        raise ValueError(f"Expected 25 outer matches, found {len(matches)}")

    mode_for_key = {(budget, family): mode for mode, (budget, family) in MODE_SPECS.items()}
    selected_outer = [
        row for row in outer_rows
        if (int(row["budget_minutes_per_match"]), row["policy_family"]) in mode_for_key
    ]
    selected_goals = [
        row for row in goal_rows
        if (int(row["budget_minutes_per_match"]), row["policy_family"]) in mode_for_key
    ]
    per_match: list[dict[str, object]] = []
    for row in selected_outer:
        budget = int(row["budget_minutes_per_match"])
        family = row["policy_family"]
        per_match.append({
            "mode": mode_for_key[(budget, family)],
            "budget_minutes_per_match": budget,
            "policy_family": family,
            "match_id": row["outer_match"],
            "catalog_goals": int(row["outer_goals"]),
            "covered_goals": int(row["outer_covered_goals"]),
            "recall": f"{int(row['outer_covered_goals']) / max(1, int(row['outer_goals'])):.6f}",
            "unique_video_minutes": float(row["outer_unique_review_minutes"]),
            "merged_segments": int(row["outer_merged_segments"]),
            "selected_candidates": int(row["outer_selected_candidates"]),
            "selected_method": row["policy_selected"],
            "window_seconds": int(row["window_seconds"]),
            "k_per_match": int(row["k_per_match"]),
            "budget_exceeded": row["outer_budget_exceeded"].lower() == "true",
            "evaluation_scope": "nested_outer_fold_oof",
        })
    per_match.sort(key=lambda row: (str(row["mode"]), str(row["match_id"])))
    if any(bool(row["budget_exceeded"]) for row in per_match):
        raise AssertionError("A selected mode exceeds its held-out budget")

    oof_goals = []
    for row in selected_goals:
        budget = int(row["budget_minutes_per_match"])
        family = row["policy_family"]
        oof_goals.append({
            **row,
            "mode": mode_for_key[(budget, family)],
            "evaluation_scope": "nested_outer_fold_oof",
        })
    oof_goals.sort(key=lambda row: (str(row["mode"]), str(row["match_id"]), str(row["goal_catalog_id"])))

    # Make a like-for-like table for all family/budget points and compute the
    # Pareto frontier in the Nested protocol only.
    points = []
    for row in summary_rows:
        points.append({
            "budget_minutes_per_match": int(row["budget_minutes_per_match"]),
            "policy_family": row["policy_family"],
            "covered_goals": int(row["covered_goals"]),
            "catalog_goals": int(row["catalog_goals"]),
            "recall": float(row["recall"]),
            "mean_unique_video_minutes": float(row["mean_unique_minutes_per_match"]),
            "max_outer_match_minutes": float(row["max_outer_match_minutes"]),
            "mean_segments_per_match": float(row["mean_merged_segments_per_match"]),
            "evaluation_scope": "nested_outer_fold_oof",
        })
    for point in points:
        point["pareto_efficient"] = not any(
            other is not point
            and other["recall"] >= point["recall"]
            and other["mean_unique_video_minutes"] <= point["mean_unique_video_minutes"]
            and (
                other["recall"] > point["recall"]
                or other["mean_unique_video_minutes"] < point["mean_unique_video_minutes"]
            )
            for other in points
        )

    # Bootstrap by match, paired across policy families at the same budget.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(matches), size=(BOOTSTRAP_REPLICATES, len(matches)))
    outer_index = {
        (int(row["budget_minutes_per_match"]), row["policy_family"], row["outer_match"]): row
        for row in outer_rows
    }
    summary_index = {
        (int(row["budget_minutes_per_match"]), row["policy_family"]): row
        for row in summary_rows
    }
    bootstrap_rows: list[dict[str, object]] = []
    vector_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for budget in (30, 40, 45, 50, 58):
        for family in ("GLOBAL_BEST", "NO_RESCUE", "CORE_RESCUE"):
            rows = [outer_index[(budget, family, match)] for match in matches]
            covered = np.asarray([int(row["outer_covered_goals"]) for row in rows], dtype=np.float64)
            goals = np.asarray([int(row["outer_goals"]) for row in rows], dtype=np.float64)
            minutes = np.asarray([float(row["outer_unique_review_minutes"]) for row in rows], dtype=np.float64)
            segments = np.asarray([int(row["outer_merged_segments"]) for row in rows], dtype=np.float64)
            vector_cache[(budget, family)] = (covered, goals, minutes)
            sampled_recall = covered[samples].sum(axis=1) / goals[samples].sum(axis=1)
            sampled_minutes = minutes[samples].mean(axis=1)
            sampled_segments = segments[samples].mean(axis=1)
            point_summary = next(
                row for row in summary_rows
                if int(row["budget_minutes_per_match"]) == budget and row["policy_family"] == family
            )
            for metric, values, observed in (
                ("recall", sampled_recall, float(point_summary["recall"])),
                ("mean_unique_video_minutes", sampled_minutes, float(point_summary["mean_unique_minutes_per_match"])),
                ("mean_merged_segments", sampled_segments, float(point_summary["mean_merged_segments_per_match"])),
            ):
                low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
                bootstrap_rows.append({
                    "budget_minutes_per_match": budget,
                    "policy_family": family,
                    "comparison": "point_estimate",
                    "metric": metric,
                    "observed_point_estimate": f"{observed:.6f}",
                    "bootstrap_p025": f"{low:.6f}",
                    "bootstrap_median": f"{median:.6f}",
                    "bootstrap_p975": f"{high:.6f}",
                    "bootstrap_probability_positive_delta": "",
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                    "resampling_unit": "match_id; paired same draws across policies",
                    "interpretation": "descriptive TRAIN outer-fold stability interval; not formal superiority proof",
                })
        for left, right in (("CORE_RESCUE", "NO_RESCUE"), ("GLOBAL_BEST", "NO_RESCUE")):
            left_cover, left_goals, left_minutes = vector_cache[(budget, left)]
            right_cover, right_goals, right_minutes = vector_cache[(budget, right)]
            delta_recall = left_cover[samples].sum(axis=1) / left_goals[samples].sum(axis=1) - right_cover[samples].sum(axis=1) / right_goals[samples].sum(axis=1)
            delta_minutes = left_minutes[samples].mean(axis=1) - right_minutes[samples].mean(axis=1)
            left_summary = summary_index[(budget, left)]
            right_summary = summary_index[(budget, right)]
            observed_delta_recall = float(left_summary["recall"]) - float(right_summary["recall"])
            observed_delta_minutes = float(left_summary["mean_unique_minutes_per_match"]) - float(right_summary["mean_unique_minutes_per_match"])
            for metric, values in (("delta_recall", delta_recall), ("delta_mean_unique_video_minutes", delta_minutes)):
                low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
                observed = observed_delta_recall if metric == "delta_recall" else observed_delta_minutes
                bootstrap_rows.append({
                    "budget_minutes_per_match": budget,
                    "policy_family": f"{left}_minus_{right}",
                    "comparison": "paired_difference",
                    "metric": metric,
                    "observed_point_estimate": f"{observed:.6f}",
                    "bootstrap_p025": f"{low:.6f}",
                    "bootstrap_median": f"{median:.6f}",
                    "bootstrap_p975": f"{high:.6f}",
                    "bootstrap_probability_positive_delta": f"{np.mean(values > 0):.6f}",
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                    "resampling_unit": "match_id; same sampled matches for both families",
                    "interpretation": "descriptive paired TRAIN outer-fold stability; not formal superiority proof",
                })

    write_csv(output / "nested_mode_per_match_oof_v1.csv", per_match)
    write_csv(output / "nested_mode_goal_coverage_oof_v1.csv", oof_goals)
    write_csv(output / "nested_pareto_points_oof_v1.csv", points)
    write_csv(output / "nested_match_bootstrap_v1.csv", bootstrap_rows)
    manifest = {
        "experiment_id": "REPORT94-FULLPOOL-CONSOLIDATION-V1",
        "scope": "TRAIN-only post-processing of 25-match nested outer-fold outputs; no model or policy retuning",
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "rows": len(read_csv(path))}
            for name, path in {
                "summary": INPUT / "nested_time_budget_summary_v1.csv",
                "outer_folds": INPUT / "nested_time_budget_outer_folds_v1.csv",
                "goal_coverage": INPUT / "nested_time_budget_goal_coverage_v1.csv",
            }.items()
        },
        "mode_mapping": {mode: {"budget_minutes_per_match": budget, "policy_family": family} for mode, (budget, family) in MODE_SPECS.items()},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "resampling_unit": "match_id", "paired": True},
        "pareto_scope": "only the 15 nested family x budget points; older exploratory points excluded",
        "interpretation": "bootstrap intervals describe between-match uncertainty in TRAIN nested outer-fold outputs; full-TRAIN final-mode selection is a separate in-sample config-selection step",
        "limits": [
            "these estimates apply to nested family/budget selection procedures, not the exact fixed config selected on all TRAIN",
            "bootstrap does not capture all training-set, model or policy-selection uncertainty",
            "intervals are descriptive and not a formal superiority test",
        ],
    }
    (output / "summary_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"matches={len(matches)} per_match_rows={len(per_match)} goal_rows={len(oof_goals)} pareto_points={len(points)} bootstrap_rows={len(bootstrap_rows)}")
    print(f"bootstrap_replicates={BOOTSTRAP_REPLICATES} seed={BOOTSTRAP_SEED}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
