"""Select fixed NO_RESCUE modes from fully held-out TRAIN LOMO predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

import evaluate_nested_time_budget_lomo_fullpool_v1 as nested


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3"
SELECTION_METHODS = ("TEMPORAL", "RRF3", "RRF4_NET")
MODE_BUDGETS = (("FAST_V3", 30), ("BALANCED_V3", 40), ("HIGH_RECALL_V3", 58))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_score(model, features: np.ndarray, labels: np.ndarray, train_mask: np.ndarray, predict_x: np.ndarray) -> np.ndarray:
    fitted = model.fit(features[train_mask], labels[train_mask])
    positive_index = list(fitted.classes_).index(1)
    return fitted.predict_proba(predict_x)[:, positive_index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    checkpoint_path = output / "fixed_policy_lomo_checkpoint_v3.csv"

    paths = {
        "temporal_features": ROOT / "results/local_trajectory_dataset_train_fullpool_v1/temporal_group_dataset_local_trajectory_v1.csv",
        "net_features": ROOT / "results/net_motion_dataset_train_fullpool_nested_v1/temporal_group_dataset_net_motion_v1.csv",
        "candidate_scores": ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv",
        "candidate_embeddings": ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.npy",
        "candidate_embedding_manifest": ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.json",
        "verifier_predictions": ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/predictions_lomo_v1.csv",
        "verifier_embeddings": ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/sequence_embeddings_v1.npz",
        "goal_catalog": ROOT / "results/local_trajectory_dataset_train_fullpool_v1/goal_timestamps_train_subset_v1.csv",
        "frozen_temporal_model": ROOT / "experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    temporal, temporal_fields = read_csv(paths["temporal_features"])
    net, net_fields = read_csv(paths["net_features"])
    candidate_rows, _ = read_csv(paths["candidate_scores"])
    verifier_rows, _ = read_csv(paths["verifier_predictions"])
    goals_rows, _ = read_csv(paths["goal_catalog"])
    if any(row.get("split", "train").lower() != "train" for row in goals_rows):
        raise AssertionError("LOMO mode selection accepts TRAIN catalog goals only")

    temporal_by_key = {(row["match_id"], row["group_uid"]): row for row in temporal}
    net_by_key = {(row["match_id"], row["group_uid"]): row for row in net}
    candidates_by_key = {(row["match_id"], row["group_uid"]): row for row in candidate_rows}
    common = set(temporal_by_key) & set(net_by_key) & set(candidates_by_key)
    if len(common) != 5742:
        raise ValueError(f"Expected 5,742 common groups; found {len(common)}")
    keys = sorted(common)
    matches = sorted({key[0] for key in keys})
    if len(matches) != 25:
        raise ValueError(f"Expected 25 TRAIN matches; found {len(matches)}")

    frozen = joblib.load(paths["frozen_temporal_model"])
    temporal_features = list(frozen["features"])
    excluded = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    net_features = [field for field in net_fields if field not in excluded]
    candidate_ids = {row["candidate_id"]: index for index, row in enumerate(candidate_rows)}
    candidate_archive = np.load(paths["candidate_embeddings"], mmap_mode="r", allow_pickle=False)
    candidate_manifest = json.loads(paths["candidate_embedding_manifest"].read_text(encoding="utf-8"))
    if candidate_ids != {value: index for index, value in enumerate(candidate_manifest["candidate_ids"])}:
        raise ValueError("Candidate embeddings manifest does not match candidate scores")
    if candidate_archive.shape != (len(candidate_rows), 4096):
        raise ValueError(f"Unexpected candidate embedding shape: {candidate_archive.shape}")
    candidate_x_by_key = {
        key: candidate_archive[candidate_ids[candidates_by_key[key]["candidate_id"]]]
        for key in keys
    }

    verifier_archive = np.load(paths["verifier_embeddings"], allow_pickle=False)
    verifier_by_id = {
        str(clip_id): embedding.reshape(-1)
        for clip_id, embedding in zip(verifier_archive["clip_ids"], verifier_archive["embeddings"])
    }
    verifier_x = np.stack([verifier_by_id[row["clip_id"]] for row in verifier_rows])
    verifier_y = np.asarray([int(row["label"] == "GOAL") for row in verifier_rows], dtype=np.int32)
    verifier_matches = np.asarray([row["match_id"] for row in verifier_rows])
    if len(verifier_rows) != 304 or set(verifier_y) != {0, 1} or len(set(verifier_matches)) != 25:
        raise ValueError("Unexpected verifier TRAIN data")

    temporal_x = np.asarray([[float(temporal_by_key[k][f]) for f in temporal_features] for k in keys], dtype=np.float32)
    net_x = np.asarray([[float(net_by_key[k][f]) for f in net_features] for k in keys], dtype=np.float32)
    temporal_y = np.asarray([int(temporal_by_key[k]["weak_label"]) for k in keys], dtype=np.int32)
    net_y = np.asarray([int(net_by_key[k]["weak_label"]) for k in keys], dtype=np.int32)
    if not np.array_equal(temporal_y, net_y):
        raise ValueError("Temporal/net labels differ on the common pool")
    key_index = {key: i for i, key in enumerate(keys)}
    candidate_by_key = {}
    for key in keys:
        candidate = candidates_by_key[key]
        candidate_by_key[key] = {
            **candidate,
            "group_uid": key[1],
            "match_id": key[0],
            "catalog_goal_in_clip": int(candidate["catalog_goal_in_clip"]),
        }
    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in goals_rows:
        if row["annotation_status"] == "completed" and row.get("event_timestamp_seconds"):
            goals[row["match_id"]].append(row)
    if sum(len(values) for values in goals.values()) != 152 or set(goals) != set(matches):
        raise ValueError("Expected exactly 152 completed TRAIN goal timestamps")

    output.mkdir(parents=True, exist_ok=True)
    # Each held match is absent from the three base estimators and the learned
    # fusion model. Development fusion features are themselves generated by
    # leaving both the outer and each inner match out of the base estimators.
    oof_rows: list[dict[str, object]] = []
    variants = [(method, window, k) for method in SELECTION_METHODS for window in nested.WINDOWS for k in nested.K_VALUES]
    completed_matches: set[str] = set()
    if checkpoint_path.is_file():
        checkpoint_rows, _ = read_csv(checkpoint_path)
        oof_rows.extend(checkpoint_rows)
        completed_matches = {str(row["match_id"]) for row in checkpoint_rows}
        if any(sum(str(row["match_id"]) == match for row in checkpoint_rows) != len(variants) for match in completed_matches):
            raise ValueError("LOMO checkpoint contains an incomplete match fold")
        print(f"resume_checkpoint_matches={len(completed_matches)}/25", flush=True)
    for fold_index, held_match in enumerate(matches, 1):
        if held_match in completed_matches:
            continue
        dev_matches = [match for match in matches if match != held_match]
        dev_rows: list[dict[str, object]] = []
        for inner_match in dev_matches:
            train_mask = np.asarray([key[0] not in {held_match, inner_match} for key in keys])
            inner_keys = [key for key in keys if key[0] == inner_match]
            inner_indices = np.asarray([key_index[key] for key in inner_keys], dtype=int)
            temporal_score = fit_score(nested.rf(), temporal_x, temporal_y, train_mask, temporal_x[inner_indices])
            net_score = fit_score(nested.rf(), net_x, net_y, train_mask, net_x[inner_indices])
            verifier_mask = (verifier_matches != held_match) & (verifier_matches != inner_match)
            visual_model = nested.lr().fit(verifier_x[verifier_mask], verifier_y[verifier_mask])
            candidate_x = np.stack([candidate_x_by_key[key] for key in inner_keys])
            visual_score = visual_model.predict_proba(candidate_x)[:, 1]
            for i, key in enumerate(inner_keys):
                dev_rows.append({
                    **candidate_by_key[key],
                    "rank_score": float(temporal_score[i]),
                    "visual_score_goal": float(visual_score[i]),
                    "net_motion": float(net_score[i]),
                })

        fusion_x = np.asarray([[float(row["rank_score"]), float(row["visual_score_goal"])] for row in dev_rows], dtype=np.float64)
        fusion_y = np.asarray([int(row["catalog_goal_in_clip"]) for row in dev_rows], dtype=np.int32)
        fusion_model = nested.lr().fit(fusion_x, fusion_y)

        outer_mask = np.asarray([key[0] != held_match for key in keys])
        held_keys = [key for key in keys if key[0] == held_match]
        held_indices = np.asarray([key_index[key] for key in held_keys], dtype=int)
        temporal_score = fit_score(nested.rf(), temporal_x, temporal_y, outer_mask, temporal_x[held_indices])
        net_score = fit_score(nested.rf(), net_x, net_y, outer_mask, net_x[held_indices])
        verifier_mask = verifier_matches != held_match
        visual_model = nested.lr().fit(verifier_x[verifier_mask], verifier_y[verifier_mask])
        candidate_x = np.stack([candidate_x_by_key[key] for key in held_keys])
        visual_score = visual_model.predict_proba(candidate_x)[:, 1]
        outer_fusion = fusion_model.predict_proba(np.column_stack([temporal_score, visual_score]))[:, 1]
        outer_rows = []
        for i, key in enumerate(held_keys):
            outer_rows.append({
                **candidate_by_key[key],
                "rank_score": float(temporal_score[i]),
                "visual_score_goal": float(visual_score[i]),
                "fusion_score_goal": float(outer_fusion[i]),
                "net_motion": float(net_score[i]),
            })

        for method, window, k in variants:
            delivered = nested.delivery_rows(outer_rows, window)
            selected = nested.select_policy(delivered, method, k)
            segment_count, unique_minutes = nested.cost_minutes(selected)
            intervals = [(float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in selected]
            covered_ids = [
                goal["goal_catalog_id"] for goal in goals[held_match]
                if any(start <= float(goal["event_timestamp_seconds"]) <= end for start, end in intervals)
            ]
            oof_rows.append({
                "match_id": held_match,
                "method": method,
                "window_seconds": window,
                "k_per_match": k,
                "covered_goals": len(covered_ids),
                "covered_goal_ids_json": json.dumps(covered_ids, ensure_ascii=False),
                "catalog_goals": len(goals[held_match]),
                "selected_candidates": len(selected),
                "merged_segments": segment_count,
                "unique_video_minutes": f"{unique_minutes:.6f}",
                "metric_role": "outer_match_lomo_cv_configuration_selection_not_nested_performance_estimate",
            })
        write_csv(checkpoint_path, oof_rows)
        if fold_index % args.progress_every == 0 or fold_index == len(matches):
            print(f"lomo_fold={fold_index}/25 held_match={held_match}", flush=True)

    aggregate_rows: list[dict[str, object]] = []
    for method, window, k in variants:
        rows = [r for r in oof_rows if r["method"] == method and int(r["window_seconds"]) == window and int(r["k_per_match"]) == k]
        total_goals = sum(int(r["catalog_goals"]) for r in rows)
        total_covered = sum(int(r["covered_goals"]) for r in rows)
        minutes = [float(r["unique_video_minutes"]) for r in rows]
        aggregate_rows.append({
            "method": method,
            "window_seconds": window,
            "k_per_match": k,
            "covered_goals": total_covered,
            "catalog_goals": total_goals,
            "recall": f"{total_covered / total_goals:.6f}",
            "mean_unique_video_minutes": f"{sum(minutes) / len(minutes):.6f}",
            "max_match_unique_video_minutes": f"{max(minutes):.6f}",
            "mean_merged_segments_per_match": f"{sum(int(r['merged_segments']) for r in rows) / len(rows):.6f}",
            "mean_selected_candidates_per_match": f"{sum(int(r['selected_candidates']) for r in rows) / len(rows):.6f}",
            "matches": len(rows),
            "selection_basis": "25-fold LOMO outer predictions; this grid-wide selection is not an independent performance estimate",
        })

    # Build a non-duplicated operating ladder. A slower tier is emitted only if
    # it is a distinct policy and improves the previous tier's LOMO coverage.
    selected_modes: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    for mode_name, budget in MODE_BUDGETS:
        feasible = [r for r in aggregate_rows if float(r["max_match_unique_video_minutes"]) <= budget + 1e-9]
        if previous is not None:
            feasible = [
                r for r in feasible
                if (r["method"], r["window_seconds"], r["k_per_match"]) != (previous["method"], previous["window_seconds"], previous["k_per_match"])
                and int(r["covered_goals"]) > int(previous["lomo_covered_goals_for_selection"])
            ]
        if not feasible:
            print(f"mode={mode_name} omitted: no distinct LOMO policy improves prior tier under {budget} minutes", flush=True)
            continue
        best = max(feasible, key=lambda r: nested.policy_key(
            {"covered_goals": int(r["covered_goals"]), "mean_unique_minutes_per_match": float(r["mean_unique_video_minutes"]), "merged_segments": round(float(r["mean_merged_segments_per_match"]) * 25)},
            str(r["method"]), int(r["window_seconds"]), int(r["k_per_match"]),
        ))
        selected = {
            "mode": mode_name,
            "budget_minutes_per_match": budget,
            "method": best["method"],
            "window_seconds": best["window_seconds"],
            "k_per_match": best["k_per_match"],
            "gap_seconds": nested.GAP_SECONDS,
            "lomo_covered_goals_for_selection": best["covered_goals"],
            "lomo_catalog_goals": best["catalog_goals"],
            "lomo_selection_recall": best["recall"],
            "lomo_mean_unique_minutes": best["mean_unique_video_minutes"],
            "lomo_max_match_unique_minutes": best["max_match_unique_video_minutes"],
            "lomo_mean_merged_segments": best["mean_merged_segments_per_match"],
            "lomo_mean_selected_candidates": best["mean_selected_candidates_per_match"],
            "metric_role": "LOMO_CV_USED_FOR_CONFIGURATION_SELECTION_NOT_UNBIASED_PERFORMANCE_ESTIMATE",
        }
        selected_modes.append(selected)
        previous = selected

    write_csv(output / "fixed_policy_lomo_by_match_v3.csv", oof_rows)
    write_csv(output / "fixed_policy_lomo_summary_v3.csv", aggregate_rows)
    if selected_modes:
        write_csv(output / "final_mode_selection_v3.csv", selected_modes)
    (output / "selection_manifest_v3.json").write_text(json.dumps({
        "experiment_id": "REPORT98-NO-RESCUE-MODES-LOMO-V3",
        "scope": "TRAIN-only; all 25 held matches scored out of fold; VALIDATION and TEST not read",
        "population": {"matches": 25, "candidate_groups": 5742, "catalog_goals": 152},
        "methods": list(SELECTION_METHODS),
        "windows_seconds": list(nested.WINDOWS),
        "k_per_match": list(nested.K_VALUES),
        "gap_seconds": nested.GAP_SECONDS,
        "mode_budget_tiers": dict(MODE_BUDGETS),
        "selection_rule": "within each tier, require max LOMO minutes within cap; maximize covered catalog goals, then minimize mean unique minutes, merged segments, and frozen complexity order; emit next tier only if it has a distinct policy and strictly higher LOMO coverage",
        "methodological_note": "LOMO predictions are out of fold for every held match; searching 72 fixed configurations on these same OOF outcomes is still configuration selection, so its winning coverage is not an unbiased performance estimate. The separate 25-fold Nested result estimates the policy-selection procedure.",
        "models": "Each outer match excluded from temporal/net RandomForest, visual verifier and fusion fitting. Fusion training uses only cross-fitted base scores from the 24 development matches.",
        "inputs_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "selected_mode_count": len(selected_modes),
        "selected_modes": selected_modes,
        "test_policy": "TEST has not been opened or used for model inference; freeze V3 before TEST evaluation",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_modes": selected_modes, "mode_count": len(selected_modes), "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
