"""Select fixed operational modes from the frozen TRAIN grid.

Full-TRAIN fit scores are used only to choose a concrete configuration from
the already frozen grid. They are not reported as generalization estimates;
those remain the nested LOMO outer-fold results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

import evaluate_nested_time_budget_lomo_v1 as nested


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments/ball/REPORT90-FINAL-MODES-EXP-0001"
MODE_SPECS = {
    "FAST_V1": {"budget": 30, "family": "NO_RESCUE"},
    "BALANCED_V1": {"budget": 40, "family": "CORE_RESCUE"},
    "HIGH_RECALL_V1": {"budget": 58, "family": "CORE_RESCUE"},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
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
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)

    paths = {
        "temporal_features": ROOT / "results/local_trajectory_dataset_train_fullpool_v1/temporal_group_dataset_local_trajectory_v1.csv",
        "net_features": ROOT / "results/net_motion_dataset_train_v1/temporal_group_dataset_net_motion_v1.csv",
        "candidate_scores": ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv",
        "candidate_embeddings": ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.npy",
        "candidate_embedding_manifest": ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.json",
        "verifier_predictions": ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/predictions_lomo_v1.csv",
        "verifier_embeddings": ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/sequence_embeddings_v1.npz",
        "goal_catalog": ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv",
        "frozen_temporal_model": ROOT / "experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    temporal, temporal_fields = nested.read_csv(paths["temporal_features"])
    net, net_fields = nested.read_csv(paths["net_features"])
    candidate_rows, _ = nested.read_csv(paths["candidate_scores"])
    verifier_rows, _ = nested.read_csv(paths["verifier_predictions"])
    goals_rows, _ = nested.read_csv(paths["goal_catalog"])
    if any(row.get("split", "train").lower() != "train" for row in goals_rows):
        raise AssertionError("Final mode selection accepts TRAIN catalog goals only")

    temporal_by_key = {(row["match_id"], row["group_uid"]): row for row in temporal}
    net_by_key = {(row["match_id"], row["group_uid"]): row for row in net}
    candidate_by_key = {(row["match_id"], row["group_uid"]): row for row in candidate_rows}
    common = set(temporal_by_key) & set(net_by_key) & set(candidate_by_key)
    if len(common) != 5372:
        raise ValueError(f"Expected frozen TRAIN population of 5,372 groups; found {len(common)}")
    keys = sorted(common)
    matches = sorted({key[0] for key in keys})
    if len(matches) != 22:
        raise ValueError(f"Expected 22 TRAIN matches; found {len(matches)}")

    frozen = joblib.load(paths["frozen_temporal_model"])
    temporal_features = list(frozen["features"])
    excluded = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    net_features = [field for field in net_fields if field not in excluded]
    temporal_x = np.asarray(
        [[float(temporal_by_key[key][field]) for field in temporal_features] for key in keys], dtype=np.float32
    )
    net_x = np.asarray(
        [[float(net_by_key[key][field]) for field in net_features] for key in keys], dtype=np.float32
    )
    temporal_y = np.asarray([int(temporal_by_key[key]["weak_label"]) for key in keys], dtype=np.int32)
    net_y = np.asarray([int(net_by_key[key]["weak_label"]) for key in keys], dtype=np.int32)
    if not np.array_equal(temporal_y, net_y):
        raise ValueError("Temporal and net-motion labels differ on the frozen common pool")

    candidate_manifest = json.loads(paths["candidate_embedding_manifest"].read_text(encoding="utf-8"))
    candidate_embeddings = np.load(paths["candidate_embeddings"], mmap_mode="r", allow_pickle=False)
    candidate_index = {str(value): i for i, value in enumerate(candidate_manifest["candidate_ids"])}
    if len(candidate_index) != len(candidate_rows) or candidate_embeddings.shape != (len(candidate_rows), 4096):
        raise ValueError("Candidate embeddings, manifest and score rows do not align")
    candidate_embeddings_by_key = {
        key: candidate_embeddings[candidate_index[str(candidate_by_key[key]["candidate_id"])]]
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
    if len(verifier_rows) != 304 or set(verifier_y) != {0, 1}:
        raise ValueError("Unexpected frozen TRAIN verifier examples")

    catalog: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in goals_rows:
        if row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds"):
            catalog[row["match_id"]].append(row)
    if sum(map(len, catalog.values())) != 130 or set(catalog) != set(matches):
        raise ValueError("Expected exactly 130 completed TRAIN goals over the 22 TRAIN matches")

    # Cross-fitted base scores train the fusion layer without leaking a match
    # into its own temporal, visual or net-motion score.
    oof_base: dict[tuple[str, str], dict[str, float]] = {}
    match_array = np.asarray([key[0] for key in keys])
    key_to_index = {key: i for i, key in enumerate(keys)}
    for index, held_match in enumerate(matches, 1):
        train_mask = match_array != held_match
        held_keys = [key for key in keys if key[0] == held_match]
        held_indices = np.asarray([key_to_index[key] for key in held_keys], dtype=int)
        temporal_model = nested.rf().fit(temporal_x[train_mask], temporal_y[train_mask])
        net_model = nested.rf().fit(net_x[train_mask], net_y[train_mask])
        verifier_mask = verifier_matches != held_match
        visual_model = nested.lr().fit(verifier_x[verifier_mask], verifier_y[verifier_mask])
        held_embeddings = np.stack([candidate_embeddings_by_key[key] for key in held_keys])
        temporal_scores = temporal_model.predict_proba(temporal_x[held_indices])[:, list(temporal_model.classes_).index(1)]
        net_scores = net_model.predict_proba(net_x[held_indices])[:, list(net_model.classes_).index(1)]
        visual_scores = visual_model.predict_proba(held_embeddings)[:, 1]
        for row_index, key in enumerate(held_keys):
            oof_base[key] = {
                "rank_score_oof": float(temporal_scores[row_index]),
                "visual_score_oof": float(visual_scores[row_index]),
                "net_motion_score_oof": float(net_scores[row_index]),
            }
        print(f"oof_base_fold={index}/{len(matches)} held_match={held_match}", flush=True)

    oof_fusion_rows = [
        [oof_base[key]["rank_score_oof"], oof_base[key]["visual_score_oof"]]
        for key in keys
    ]
    fusion_y = np.asarray([int(candidate_by_key[key]["catalog_goal_in_clip"]) for key in keys], dtype=np.int32)
    fusion_model = nested.lr().fit(np.asarray(oof_fusion_rows, dtype=np.float64), fusion_y)

    # Fit the base estimators on all TRAIN for the operational frozen bundle.
    # Their scores on TRAIN are only used to choose from the closed grid.
    temporal_model = nested.rf().fit(temporal_x, temporal_y)
    net_model = nested.rf().fit(net_x, net_y)
    visual_model = nested.lr().fit(verifier_x, verifier_y)
    temporal_all = temporal_model.predict_proba(temporal_x)[:, list(temporal_model.classes_).index(1)]
    net_all = net_model.predict_proba(net_x)[:, list(net_model.classes_).index(1)]
    visual_all = visual_model.predict_proba(
        np.stack([candidate_embeddings_by_key[key] for key in keys])
    )[:, 1]
    fusion_all = fusion_model.predict_proba(np.column_stack([temporal_all, visual_all]))[:, 1]

    selection_rows_by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    base_scores_rows: list[dict[str, object]] = []
    for index, key in enumerate(keys):
        candidate = candidate_by_key[key]
        row: dict[str, object] = {
            **candidate,
            "group_uid": key[1],
            "match_id": key[0],
            "catalog_goal_in_clip": int(candidate["catalog_goal_in_clip"]),
            "rank_score": float(temporal_all[index]),
            "visual_score_goal": float(visual_all[index]),
            "fusion_score_goal": float(fusion_all[index]),
            "net_motion": float(net_all[index]),
        }
        selection_rows_by_match[key[0]].append(row)
        base_scores_rows.append({
            "match_id": key[0], "group_uid": key[1],
            **oof_base[key],
            "rank_score_full_train": f"{temporal_all[index]:.9f}",
            "visual_score_full_train": f"{visual_all[index]:.9f}",
            "fusion_score_full_train": f"{fusion_all[index]:.9f}",
            "net_motion_score_full_train": f"{net_all[index]:.9f}",
        })

    mode_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    for mode_name, spec in MODE_SPECS.items():
        budget = int(spec["budget"])
        family = str(spec["family"])
        family_methods = nested.POLICY_FAMILIES[family]
        feasible: list[tuple[tuple[str, int, int], dict[str, object]]] = []
        for method in family_methods:
            for window in nested.WINDOWS:
                for k in nested.K_VALUES:
                    result = nested.score_policy(selection_rows_by_match, catalog, method, k, window)
                    is_feasible = float(result["max_unique_minutes_per_match"]) <= budget + 1e-9
                    grid_row = {
                        "mode": mode_name, "family": family, "budget_minutes_per_match": budget,
                        **result, "feasible_on_full_train_selection_data": is_feasible,
                        "metric_role": "in_sample_selection_only",
                    }
                    grid_rows.append(grid_row)
                    if is_feasible:
                        feasible.append(((method, window, k), result))
        if not feasible:
            raise RuntimeError(f"No feasible fixed configuration for {mode_name}")
        (method, window, k), result = max(
            feasible,
            key=lambda item: nested.policy_key(item[1], item[0][0], item[0][1], item[0][2]),
        )
        mode_rows.append({
            "mode": mode_name,
            "budget_minutes_per_match": budget,
            "selection_family": family,
            "method": method,
            "window_seconds": window,
            "k_per_match": k,
            "gap_seconds": nested.GAP_SECONDS,
            "training_selection_covered_goals": result["covered_goals"],
            "training_selection_catalog_goals": result["catalog_goals"],
            "training_selection_recall": f"{float(result['recall']):.6f}",
            "training_selection_mean_unique_minutes": f"{float(result['mean_unique_minutes_per_match']):.3f}",
            "training_selection_max_unique_minutes": f"{float(result['max_unique_minutes_per_match']):.3f}",
            "training_selection_merged_segments": result["merged_segments"],
            "metric_role": "in_sample_selection_only_not_generalization_estimate",
        })

    model_paths = {
        "temporal_ranker": models_dir / "temporal_ranker.joblib",
        "net_motion_ranker": models_dir / "net_motion_ranker.joblib",
        "visual_verifier": models_dir / "visual_verifier.joblib",
        "fusion_ranker": models_dir / "fusion_ranker.joblib",
    }
    joblib.dump({"model": temporal_model, "features": temporal_features}, model_paths["temporal_ranker"])
    joblib.dump({"model": net_model, "features": net_features}, model_paths["net_motion_ranker"])
    joblib.dump(visual_model, model_paths["visual_verifier"])
    joblib.dump({"model": fusion_model, "features": ["rank_score", "visual_score_goal"]}, model_paths["fusion_ranker"])
    model_manifest = {
        name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for name, path in model_paths.items()
    }
    for mode in mode_rows:
        mode["models_manifest"] = "model_manifest_v1.json"
        mode["rescue_enabled"] = mode["method"] in {"CORE35_FUSION10_NET10", "CORE40_RRF4_NET_RESCUE"}
        mode["ranking_sources"] = {
            "TEMPORAL": ["temporal_ranker"],
            "RRF3": ["temporal_ranker", "visual_verifier", "fusion_ranker"],
            "RRF4_NET": ["temporal_ranker", "visual_verifier", "fusion_ranker", "net_motion_ranker"],
            "CORE35_FUSION10_NET10": ["temporal_ranker", "fusion_ranker", "net_motion_ranker"],
            "CORE40_RRF4_NET_RESCUE": ["temporal_ranker", "visual_verifier", "fusion_ranker", "net_motion_ranker"],
        }[str(mode["method"])]
    config = {
        "config_id": "REPORT90-FINAL-MODES-V1",
        "status": "TRAIN_SELECTED_CANDIDATES_NOT_YET_FINAL_FREEZE",
        "scope": "TRAIN only; fixed mode settings selected from frozen grid; TEST and VALIDATION not accessed",
        "selection_metric_warning": "Full-TRAIN predictions are in-sample and are only used to choose a concrete setting from the pre-existing grid; use nested outer-fold results for performance estimates.",
        "budgets_are": "unique selected video minutes per match; not human review time",
        "grid": {
            "methods": list(nested.METHODS), "budgets": list(nested.BUDGETS),
            "windows_seconds": list(nested.WINDOWS), "k_per_match": list(nested.K_VALUES),
            "gap_seconds": nested.GAP_SECONDS,
        },
        "modes": {str(row["mode"]): {k: v for k, v in row.items() if k not in {"mode", "metric_role"}} for row in mode_rows},
        "model_files": model_manifest,
        "mode_results_file": "final_mode_selection_v1.csv",
        "full_train_grid_file": "full_train_mode_grid_v1.csv",
        "score_diagnostic_file": "base_scores_oof_and_full_train_v1.csv",
    }
    write_csv(output / "final_mode_selection_v1.csv", mode_rows)
    write_csv(output / "full_train_mode_grid_v1.csv", grid_rows)
    write_csv(output / "base_scores_oof_and_full_train_v1.csv", base_scores_rows)
    write_json(output / "model_manifest_v1.json", model_manifest)
    write_json(output / "final_modes_v1.json", config)
    write_json(output / "run_manifest_v1.json", {
        "experiment_id": "REPORT90-FINAL-MODES-EXP-0001",
        "scope": config["scope"],
        "selection_is_in_sample": True,
        "generalization_estimates_source": "NESTED-TIME-BUDGET-LOMO-EXP-0001 and NESTED-TIME-BUDGET-FAMILY-COMPARISON-V1",
        "matches": len(matches), "candidate_groups": len(keys),
        "catalog_goals": sum(map(len, catalog.values())),
        "oof_base_score_fit": "leave-one-match-out; held match excluded from temporal/net/visual base estimators",
        "fusion_fit": "logistic regression fit on cross-fitted temporal+visual scores and TRAIN candidate labels",
        "full_train_models": "temporal/net random forests and visual verifier refit on all TRAIN; used for operational bundle and in-sample configuration selection only",
        "tie_break": "maximize covered catalog goals, then minimize mean unique minutes, merged segments, then fixed method/window/K complexity order",
        "inputs_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "validation_test_policy": "neither VALIDATION nor TEST was opened or used",
        "model_hashes": model_manifest,
    })
    for row in mode_rows:
        print(
            f"mode={row['mode']} budget={row['budget_minutes_per_match']} family={row['selection_family']} "
            f"method={row['method']} window={row['window_seconds']} K={row['k_per_match']} "
            f"train_selection={row['training_selection_covered_goals']}/{row['training_selection_catalog_goals']} "
            f"minutes={row['training_selection_mean_unique_minutes']}"
        )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
