"""Create a hash manifest of completed Report 87 work without declaring final sprint freeze."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments/ball/REPORT87-PD-SPRINT-INTERIM-FREEZE-20261005"
ANNOTATIONS = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/frame_annotations_v1.csv"
MISSES = ROOT / "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/positive_miss_mechanism_v2_annotated.csv"
CONTROLS_PRIOR = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/pairwise_control_audit_v2_prior50_annotated.csv"
CONTROLS_BATCH2 = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/pairwise_control_audit_v2_batch2_annotated.csv"
HARDNEG100 = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100/reviewed_hard_negative_reranking.json"
TRACKER_EXP = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001"
ANNOTATION_FREEZE = TRACKER_EXP / "annotation_freeze_v1.json"
TRACKER_SUMMARY = TRACKER_EXP / "tracker_retest_metrics_v1.json"

ARTIFACTS = [
    "docs/plans/87_relatorio_sprint_final_pd_ensemble_detector_tracking_02-10-2026.md",
    "plano_tcc_jira.md",
    "experiments/ball/freeze_report87_interim_v1.py",
    "experiments/ball/finalize_report87_sprint_v1.py",
    "experiments/ball/build_complementarity_matrix_v1.py",
    "experiments/ball/COMPLEMENTARITY-MATRIX-EXP-0001/README.md",
    "experiments/ball/COMPLEMENTARITY-MATRIX-EXP-0001/summary_v1.json",
    "experiments/ball/COMPLEMENTARITY-MATRIX-EXP-0001/method_coverage_v1.csv",
    "experiments/ball/COMPLEMENTARITY-MATRIX-EXP-0001/pairwise_complementarity_v1.csv",
    "experiments/ball/COMPLEMENTARITY-MATRIX-EXP-0001/goal_coverage_by_method_v1.csv",
    "experiments/ball/HYBRID-META-RANKING-EXP-0001/README.md",
    "experiments/ball/HYBRID-META-RANKING-EXP-0001/summary_v1.json",
    "experiments/ball/HYBRID-META-RANKING-EXP-0001/core_rescue_comparison_v1.csv",
    "experiments/ball/HYBRID-META-RANKING-EXP-0001/core_rescue_by_match_v1.csv",
    "experiments/ball/HYBRID-META-RANKING-EXP-0001/core_rescue_goal_coverage_v1.csv",
    "experiments/ball/HYBRID-META-RANKING-EXP-0001/core_rescue_selected_candidates_v1.csv",
    "experiments/ball/GOAL-EVENT-BALL-EVIDENCE-RRF-ENSEMBLE-EXP-0001/summary_v1.json",
    "experiments/ball/GOAL-EVENT-BALL-EVIDENCE-RRF-ENSEMBLE-EXP-0001/recall_at_k_comparison_v1.csv",
    "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/README.md",
    "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/summarize_human_miss_review_v1.py",
    "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/positive_miss_mechanism_v2_annotated.csv",
    "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/human_visual_review_summary_v1.json",
    "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/human_visual_review_summary_v1.md",
    "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/protocol_v1.json",
    "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/holdout_evaluation_v1/protocol_v1.json",
    "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/holdout_evaluation_v1/holdout_summary_v1.csv",
    "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/holdout_evaluation_v1/tracker_gate_decision_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/README.md",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/protocol_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/event_manifest_v1.csv",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/annotation_freeze_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/tracker_retest_metrics_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/tracker_retest_group_metrics_v1.csv",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/tracker_retest_sequence_metrics_v1.csv",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/tracker_retest_pair_metrics_v1.csv",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/tracking_scoring_frames_center_v1.csv",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/tracking_scoring_frames_full_v1.csv",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/blind_tracker_input_freeze_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/blind_tracker_run_freeze_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/ball_identity_only_v1/annotation_freeze_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/ball_identity_only_v1/tracker_retest_metrics_v1.json",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/ball_identity_only_v1/tracker_retest_group_metrics_v1.csv",
    "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/ball_identity_only_v1/tracker_retest_sequence_metrics_v1.csv",
    "experiments/ball/BALL-TRAJECTORY-ANCHOR-EXP-0001/protocol_v1.json",
    "experiments/ball/BALL-TRAJECTORY-ANCHOR-EXP-0001/blind_replay_freeze_v1.json",
    "experiments/ball/BALL-TRAJECTORY-ANCHOR-EXP-0001/scored_summary_v1.csv",
    "experiments/ball/BALL-TRAJECTORY-ANCHOR-EXP-0001/scored_frame_results_v1.csv",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/pairwise_control_audit_v2_prior50_annotated.csv",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/pairwise_control_audit_v2_batch2_annotated.csv",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100/reviewed_hard_negative_reranking.json",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100/reviewed_hard_negative_reranking.md",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100/reviewed_hard_negative_lomo_scores.csv",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100/reviewed_hard_negative_pairwise_lomo.csv",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100/reviewed_hard_negative_rerank_by_match.csv",
    "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100/reviewed_hard_negative_rerank_summary.csv",
    "docs/plans/89_relatorio_consolidacao_resultados_06-10_e_plano_fechamento_sprint.md",
    "docs/plans/90_plano_final_consolidacao_freeze_r3d_opcional_06-10-2026.md",
    "experiments/ball/evaluate_nested_time_budget_lomo_v1.py",
    "experiments/ball/NESTED-TIME-BUDGET-LOMO-EXP-0001/summary_v1.json",
    "experiments/ball/NESTED-TIME-BUDGET-LOMO-EXP-0001/nested_time_budget_summary_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-LOMO-EXP-0001/nested_time_budget_outer_folds_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-LOMO-EXP-0001/nested_time_budget_goal_coverage_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-FAMILY-COMPARISON-V1/summary_v1.json",
    "experiments/ball/NESTED-TIME-BUDGET-FAMILY-COMPARISON-V1/nested_time_budget_summary_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-FAMILY-COMPARISON-V1/nested_time_budget_outer_folds_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-FAMILY-COMPARISON-V1/nested_time_budget_goal_coverage_v1.csv",
    "experiments/ball/select_report90_final_modes_v1.py",
    "experiments/ball/REPORT90-FINAL-MODES-EXP-0001/final_modes_v1.json",
    "experiments/ball/REPORT90-FINAL-MODES-EXP-0001/final_mode_selection_v1.csv",
    "experiments/ball/REPORT90-FINAL-MODES-EXP-0001/full_train_mode_grid_v1.csv",
    "experiments/ball/REPORT90-FINAL-MODES-EXP-0001/model_manifest_v1.json",
    "experiments/ball/REPORT90-FINAL-MODES-EXP-0001/run_manifest_v1.json",
    "experiments/ball/summarize_report90_nested_oof_v1.py",
    "experiments/ball/REPORT90-CONSOLIDATION-EXP-0001/summary_v1.json",
    "experiments/ball/REPORT90-CONSOLIDATION-EXP-0001/nested_mode_per_match_oof_v1.csv",
    "experiments/ball/REPORT90-CONSOLIDATION-EXP-0001/nested_mode_goal_coverage_oof_v1.csv",
    "experiments/ball/REPORT90-CONSOLIDATION-EXP-0001/nested_pareto_points_oof_v1.csv",
    "experiments/ball/REPORT90-CONSOLIDATION-EXP-0001/nested_match_bootstrap_v1.csv",
    # Report 94: full-pool Nested extension; all files are explicit so the
    # freeze snapshot is reviewable and does not silently include temp outputs.
    "docs/plans/94_plano_execucao_nested_lomo_fullpool_25_train_14-10-2026.md",
    "docs/plans/95_resultados_nested_lomo_fullpool_25_train_14-10-2026.md",
    "experiments/ball/assemble_net_motion_dataset_fullpool_v1.py",
    "experiments/ball/evaluate_nested_time_budget_lomo_fullpool_v1.py",
    "experiments/ball/summarize_nested_fullpool_oof_v1.py",
    "experiments/ball/select_nested_fullpool_modes_v2.py",
    "experiments/ball/finalize_report94_fullpool_v2.py",
    "results/net_motion_features_train_fullpool_recovered_v1/metrics.json",
    "results/net_motion_features_train_fullpool_recovered_v1/net_motion_groups_v1.csv",
    "results/rich_temporal_dataset_train_fullpool_nested_v1/summary_v1.json",
    "results/rich_temporal_dataset_train_fullpool_nested_v1/temporal_group_dataset_v1.csv",
    "results/net_motion_dataset_train_fullpool_nested_v1/goal_timestamps_train_subset_v1.csv",
    "results/net_motion_dataset_train_fullpool_nested_v1/summary_v1.json",
    "results/net_motion_dataset_train_fullpool_nested_v1/temporal_group_dataset_net_motion_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1/summary_v1.json",
    "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1/nested_time_budget_summary_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1/nested_time_budget_outer_folds_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1/nested_time_budget_goal_coverage_v1.csv",
    "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1/nested_time_budget_policy_selection_v1.csv",
    "experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1/summary_v1.json",
    "experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1/nested_mode_per_match_oof_v1.csv",
    "experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1/nested_mode_goal_coverage_oof_v1.csv",
    "experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1/nested_pareto_points_oof_v1.csv",
    "experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1/nested_match_bootstrap_v1.csv",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/final_modes_v2.json",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/final_mode_selection_v2.csv",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/full_train_mode_grid_v2.csv",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/base_scores_oof_and_full_train_v2.csv",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/model_manifest_v2.json",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/run_manifest_v2.json",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/models/temporal_ranker.joblib",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/models/net_motion_ranker.joblib",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/models/visual_verifier.joblib",
    "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/models/fusion_ranker.joblib",
    # Report 98: fixed NO_RESCUE modes selected with held-out TRAIN LOMO.
    "docs/plans/98_plano_selecao_modos_lomo_no_rescue_v3_14-10-2026.md",
    "docs/plans/99_resultados_selecao_modos_lomo_no_rescue_v3_14-10-2026.md",
    "experiments/ball/select_no_rescue_modes_lomo_v3.py",
    "experiments/ball/materialize_report98_modes_v3.py",
    "experiments/ball/finalize_report98_modes_v3.py",
    "experiments/ball/prepare_test_goal_catalog_manifest_v3.py",
    "experiments/ball/annotate_test_goal_catalog_timestamps_v3.py",
    "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/selection_manifest_v3.json",
    "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/fixed_policy_lomo_by_match_v3.csv",
    "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/fixed_policy_lomo_summary_v3.csv",
    "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/final_mode_selection_v3.csv",
    "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/final_modes_v3.json",
    "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/run_manifest_v3.json",
    "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/model_manifest_v3.json",
    "experiments/ball/build_report90_ball_tables_v1.py",
    "experiments/ball/REPORT90-BALL-TABLES-EXP-0001/summary_v1.json",
    "experiments/ball/REPORT90-BALL-TABLES-EXP-0001/ball_proposal_availability_v1.csv",
    "experiments/ball/REPORT90-BALL-TABLES-EXP-0001/ball_tracking_identity_v1.csv",
    "experiments/ball/BALL-PROPOSAL-CEILING-EXP-0001/summary_v1.json",
    "experiments/ball/BALL-PROPOSAL-CEILING-EXP-0001/proposal_ceiling_metrics_v1.csv",
    "experiments/ball/BALL-TILED-INFERENCE-EXP-0001/summary_v1.json",
    "experiments/ball/BALL-TILED-INFERENCE-EXP-0001/tiled_inference_metrics_v1.csv",
    "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001/tiled_tracker_run_freeze_v1.json",
    "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001/tiled_tracker_scored_summary_v1.csv",
    "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001/budget_grid_v1/summary_v1.json",
    "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001/budget_grid_v1/tiled_tracker_budget_grid_summary_v1.csv",
    "experiments/ball/BALL-TILED-TRACKER-RETEST-EXP-0001/budget_grid_v1/tiled_tracker_budget_grid_by_sequence_v1.csv",
]
CHECKPOINTS = [
    "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi_raw, yi_raw) in enumerate(polygon):
        xi, yi = float(xi_raw), float(yi_raw)
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior_manifests = sorted(
        OUT_DIR.glob("interim_freeze_manifest_v*.json"),
        key=lambda path: int(path.stem.rsplit("v", 1)[1]),
    )
    manifest_version = int(prior_manifests[-1].stem.rsplit("v", 1)[1]) + 1 if prior_manifests else 1
    manifest_path = OUT_DIR / f"interim_freeze_manifest_v{manifest_version}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite interim freeze output: {manifest_path}")
    prior_manifest_names = [path.relative_to(ROOT).as_posix() for path in prior_manifests]
    snapshot_artifacts = ARTIFACTS + prior_manifest_names
    missing = [name for name in snapshot_artifacts + [str(p.relative_to(ROOT)) for p in [ANNOTATIONS, MISSES, CONTROLS_PRIOR, CONTROLS_BATCH2, HARDNEG100]] + CHECKPOINTS if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required sprint evidence:\n- " + "\n- ".join(missing))

    annotations = read_csv(ANNOTATIONS)
    if len(annotations) != 1435 or any(row["annotation_status"] != "completed" for row in annotations):
        raise ValueError("Interim freeze requires all 1,435 BALL annotations completed.")
    centers = [row for row in annotations if row["is_event_frame"] == "True"]
    missing_regions = sum(row.get("goal_region_status", "") not in {"VALID", "NOT_VISIBLE", "UNCERTAIN"} for row in centers)
    missing_volumes = sum(row.get("goal_volume_status", "") not in {"VALID", "NOT_VISIBLE", "UNCERTAIN"} for row in centers)
    if len(centers) != 35 or any(row.get("goal_region_status") != "VALID" or row.get("goal_volume_status") != "VALID" for row in centers):
        raise ValueError("Spatial freeze requires VALID region and volume annotations at all 35 event centers.")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to validate image bounds for the spatial annotation freeze.") from exc
    geometry_qa = {"event_centers": len(centers), "valid_regions": 0, "valid_volumes": 0, "points_in_bounds": 0, "volume_vertices_outside_aperture": 0}
    for row in centers:
        image_path = ROOT / row["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read annotated center frame: {image_path}")
        height, width = image.shape[:2]
        region = json.loads(row["goal_region_points_json"])
        volume = json.loads(row["goal_volume_points_json"])
        if len(region) != 4 or len(volume) != 6:
            raise ValueError(f"Expected 4 aperture and 6 volume points at {row['event_id']}.")
        for point in region + volume:
            x, y = float(point[0]), float(point[1])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Out-of-frame goal geometry at {row['event_id']}: {point}, image={width}x{height}")
            geometry_qa["points_in_bounds"] += 1
        outside = sum(not point_in_polygon((float(point[0]), float(point[1])), region) for point in volume)
        if outside < 1:
            raise ValueError(f"Projected goal volume does not extend outside the aperture at {row['event_id']}.")
        geometry_qa["valid_regions"] += 1
        geometry_qa["valid_volumes"] += 1
        geometry_qa["volume_vertices_outside_aperture"] += outside

    annotation_freeze = json.loads(ANNOTATION_FREEZE.read_text(encoding="utf-8"))
    if annotation_freeze.get("annotations_sha256") != sha256(ANNOTATIONS):
        raise ValueError("Spatial annotation freeze hash does not match the current annotation CSV.")
    if annotation_freeze.get("event_manifest_sha256") != sha256(TRACKER_EXP / "event_manifest_v1.csv"):
        raise ValueError("Spatial annotation freeze hash does not match the current event manifest.")
    tracker_summary = json.loads(TRACKER_SUMMARY.read_text(encoding="utf-8"))
    if tracker_summary.get("annotation_freeze", {}).get("sha256") != sha256(ANNOTATION_FREEZE):
        raise ValueError("Tracker scoring summary was not produced from the current spatial annotation freeze.")

    misses = read_csv(MISSES)
    if len(misses) != 127 or any(row["visual_review_status"] != "COMPLETED" for row in misses):
        raise ValueError("Interim freeze requires all 127 visual miss reviews completed.")
    prior, batch2 = read_csv(CONTROLS_PRIOR), read_csv(CONTROLS_BATCH2)
    all_controls = prior + batch2
    if len(all_controls) != 100 or any(row.get("manual_status") != "COMPLETED" or row.get("manual_label") != "NO_GOAL" for row in all_controls):
        raise ValueError("Interim freeze requires the 100 confirmed human NO_GOAL controls.")
    hardneg = json.loads(HARDNEG100.read_text(encoding="utf-8"))
    if hardneg.get("manually_confirmed_no_goal_groups") != 100:
        raise ValueError("Checkpoint100 metadata does not confirm 100 human NO_GOAL groups.")

    artifacts = {}
    for name in snapshot_artifacts + CHECKPOINTS:
        path = ROOT / name
        artifacts[name.replace("\\", "/")] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    artifacts[str(ANNOTATIONS.relative_to(ROOT)).replace("\\", "/")] = {"sha256": sha256(ANNOTATIONS), "bytes": ANNOTATIONS.stat().st_size}

    result = {
        "freeze_id": "REPORT87-PD-SPRINT-INTERIM-FREEZE-20261005",
        "manifest_version": manifest_version,
        "supersedes": prior_manifests[-1].name if prior_manifests else None,
        "is_final_sprint_freeze": False,
        "scope": "TRAIN-only evidence snapshot; TEST closed; V4 promotion not authorized",
        "planned_final_freeze_date": "2026-10-11",
        "validated_state": {
            "ball_annotations": {"completed": len(annotations), "expected": 1435},
            "positive_miss_visual_review": {"completed": len(misses), "expected": 127},
            "human_confirmed_no_goal_controls": {"completed": len(all_controls), "expected": 100},
            "goal_geometry_event_centers": {"expected": 35, "missing_region_decisions": missing_regions, "missing_volume_decisions": missing_volumes, "qa": geometry_qa},
            "full_geometry_freeze_ready": missing_regions == 0 and missing_volumes == 0 and geometry_qa["valid_regions"] == 35 and geometry_qa["valid_volumes"] == 35,
        },
        "decisions": {
            "core_rescue": "KEEP_AS_EXPLORATORY_TRAIN_SIGNAL; NO_PROMOTION",
            "detector_only": "GATE_PASSED_BY_ONE_POSITIVE_FRAME; VITERBI_RETEST_SHOWED_NO_CONSISTENT_IDENTITY_GAIN",
            "anchor_trajectory": "NO_GO_AS_TESTED",
            "hard_negative_ranker_at_100": "NO_GO_FOR_PROMOTION; STOP_AT_100_FOR_THIS_COHORT",
            "v4": "UNCHANGED; NO_PROMOTION_AUTHORIZED",
            "test": "CLOSED",
        },
        "artifacts": artifacts,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "INTERIM_FREEZE_WRITTEN", "path": str(manifest_path), "geometry_missing_regions": missing_regions, "geometry_missing_volumes": missing_volumes, "artifact_count": len(artifacts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
