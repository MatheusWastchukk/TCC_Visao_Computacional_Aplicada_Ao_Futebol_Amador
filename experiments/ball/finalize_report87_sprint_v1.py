"""Emit the final Report 87 sprint freeze after the planned 2026-10-11 date.

Run from the repository root, after creating a fresh interim snapshot:
    .\\.venv\\Scripts\\python.exe experiments\\ball\\finalize_report87_sprint_v1.py --preflight
    # on/after 2026-10-11, omit --preflight to write the final manifest
    .\\.venv\\Scripts\\python.exe experiments\\ball\\freeze_report87_interim_v1.py
    .\\.venv\\Scripts\\python.exe experiments\\ball\\finalize_report87_sprint_v1.py

This script does not change model outputs, labels, V4 or TEST. It only writes a
new immutable final hash manifest after revalidating the frozen evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT / "experiments/ball/REPORT87-PD-SPRINT-INTERIM-FREEZE-20261005"
FINAL_DIR = ROOT / "experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011"
PLANNED_DATE = date(2026, 10, 11)
ANNOTATIONS = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001/frame_annotations_v1.csv"
MISSES = ROOT / "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/positive_miss_mechanism_v2_annotated.csv"
MISS_SUMMARY = ROOT / "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002/human_visual_review_summary_v1.json"
CONTROLS_DIR = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
TRACKER_DIR = ROOT / "experiments/ball/ORACLE-TRACKER-RETEST-EXP-0001"
CORE_RESCUE = ROOT / "experiments/ball/HYBRID-META-RANKING-EXP-0001"
COMPLEMENTARITY = ROOT / "experiments/ball/COMPLEMENTARITY-MATRIX-EXP-0001"
DETECTOR_GATE = ROOT / "experiments/ball/BALL-HARD-POSITIVE-EXP-0001/holdout_evaluation_v1/tracker_gate_decision_v1.json"
HARDNEG_DIR = CONTROLS_DIR / "PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100"
ANCHOR_SUMMARY = ROOT / "experiments/ball/BALL-TRAJECTORY-ANCHOR-EXP-0001/scored_summary_v1.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen hash mismatch for {label}: expected {expected}, got {actual}")


def latest_interim_manifest() -> Path:
    manifests = sorted(
        INTERIM_DIR.glob("interim_freeze_manifest_v*.json"),
        key=lambda path: int(path.stem.rsplit("v", 1)[1]),
    )
    if not manifests:
        raise FileNotFoundError(f"No interim freeze manifest found in {INTERIM_DIR}")
    return manifests[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run all final evidence checks without the date gate or writing a final manifest.",
    )
    args = parser.parse_args()
    today = date.today()
    if today < PLANNED_DATE and not args.preflight:
        raise RuntimeError(
            f"Final freeze is scheduled for {PLANNED_DATE.isoformat()}; today is {today.isoformat()}. "
            "Use --preflight to validate readiness without writing the final manifest."
        )

    output = FINAL_DIR / "final_freeze_manifest_v1.json"
    if output.exists() and not args.preflight:
        raise FileExistsError(f"Refusing to overwrite final freeze: {output}")

    interim_path = latest_interim_manifest()
    interim = read_json(interim_path)
    if interim.get("is_final_sprint_freeze") is not False:
        raise ValueError("Expected the latest evidence snapshot to be explicitly interim.")
    if "TEST closed" not in interim.get("scope", "") or "V4 promotion not authorized" not in interim.get("scope", ""):
        raise ValueError("Interim snapshot does not preserve the TEST/V4 scope constraints.")

    artifacts = interim.get("artifacts", {})
    if not artifacts:
        raise ValueError("Interim snapshot has no artifact hashes.")
    for relative_path, metadata in artifacts.items():
        verify_file_hash(ROOT / relative_path, metadata["sha256"], relative_path)

    annotations = read_csv(ANNOTATIONS)
    if len(annotations) != 1435 or any(row.get("annotation_status") != "completed" for row in annotations):
        raise ValueError("Final freeze requires all 1,435 BALL frame annotations completed.")
    centers = [row for row in annotations if row.get("is_event_frame") == "True"]
    if len(centers) != 35 or any(
        row.get("goal_region_status") != "VALID" or row.get("goal_volume_status") != "VALID"
        for row in centers
    ):
        raise ValueError("Final freeze requires valid goal-region and goal-volume decisions at all 35 centers.")

    misses = read_csv(MISSES)
    if len(misses) != 127 or any(row.get("visual_review_status") != "COMPLETED" for row in misses):
        raise ValueError("Final freeze requires all 127 positive-miss reviews completed.")
    miss_summary = read_json(MISS_SUMMARY)
    if miss_summary.get("input_sha256") != sha256(MISSES) or miss_summary.get("completed_visual_reviews") != 127:
        raise ValueError("Positive-miss summary does not describe the current 127 reviewed rows.")
    expected_miss_tags = {
        "BAD_LOCALIZATION": 95,
        "SMALL_BALL": 54,
        "BLUR": 51,
        "LOW_CONTRAST": 31,
        "OCCLUSION_PLAYER": 15,
        "OCCLUSION_GOALKEEPER": 8,
        "OCCLUSION_NET": 0,
        "FRAME_EDGE": 1,
        "OTHER": 0,
        "UNCERTAIN": 10,
    }
    observed_miss_tags: Counter[str] = Counter()
    proposal_status_counts: Counter[str] = Counter()
    selected_box_status_counts: Counter[str] = Counter()
    frames_with_specific_visual_tag = 0
    for row in misses:
        labels = {label for label in (row.get("visual_cause_tags") or "").split(";") if label}
        unknown_labels = labels - set(expected_miss_tags)
        if unknown_labels:
            raise ValueError(f"Unknown visual miss tags on {row.get('sequence_id')}: {sorted(unknown_labels)}")
        observed_miss_tags.update(labels)
        frames_with_specific_visual_tag += bool(labels - {"UNCERTAIN"})
        proposal_status_counts[row["proposal_status"]] += 1
        selected_box_status_counts[row["selected_tracker_box_status"]] += 1
    observed_miss_tags = Counter({tag: observed_miss_tags[tag] for tag in expected_miss_tags})
    if dict(observed_miss_tags) != expected_miss_tags or miss_summary.get("tag_counts_overlapping") != expected_miss_tags:
        raise ValueError("Positive-miss visual tag counts differ from the consolidated report.")
    if (
        miss_summary.get("frames_with_uncertain_tag") != observed_miss_tags["UNCERTAIN"]
        or miss_summary.get("frames_with_at_least_one_specific_visual_tag") != frames_with_specific_visual_tag
        or miss_summary.get("objective_proposal_status") != dict(proposal_status_counts)
        or miss_summary.get("objective_selected_tracker_box_status") != dict(selected_box_status_counts)
    ):
        raise ValueError("Positive-miss objective statuses do not reconcile with the 127 annotated rows.")

    core_summary = read_json(CORE_RESCUE / "summary_v1.json")
    core_scope = core_summary.get("scope", "")
    if "22 matches" not in core_scope or "130 completed catalog goals" not in core_scope or "TRAIN only" not in core_scope:
        raise ValueError("CORE+RESCUE summary is not the expected 22-match/130-goal TRAIN evaluation.")
    core_rows = {row["variant"]: row for row in read_csv(CORE_RESCUE / "core_rescue_comparison_v1.csv")}
    expected_core = {
        "CORE10_RESCUE10_NET": (20, 52, 4, 11, 7, 0.037, 4),
        "CORE20_RESCUE10_NET": (30, 75, 3, 12, 9, -0.003, 3),
        "CORE30_RESCUE10_NET": (40, 92, 4, 8, 4, -0.008, -2),
        "CORE30_RESCUE20_NET": (50, 103, 4, 7, 3, 0.051, 8),
        "CORE35_RESCUE10_FUSION_RESCUE10_NET": (55, 111, 7, 9, 2, 0.081, 9),
        "CORE40_RESCUE10_NET": (50, 102, 3, 4, 1, 0.071, 13),
        "CORE40_RESCUE20_NET": (60, 109, 2, 6, 4, 0.090, 8),
        "CORE40_RESCUE20_RRF_NET": (60, 114, 7, 8, 1, 0.184, 16),
        "CORE60_RESCUE10_NET_RESCUE10_FUSION": (80, 118, -1, 1, 2, 0.096, 12),
    }
    if set(core_rows) != set(expected_core):
        raise ValueError("CORE+RESCUE variant set differs from the 9-row consolidated sensitivity table.")
    for variant, expected in expected_core.items():
        row = core_rows.get(variant)
        observed = (
            int(row["budget_k_per_match"]),
            int(row["covered_goals"]),
            int(row["net_goal_change_vs_baseline"]),
            int(row["goals_rescued_vs_baseline"]),
            int(row["baseline_goals_lost"]),
            round(float(row["unique_review_minutes_per_match"]) - float(row["baseline_unique_review_minutes_per_match"]), 3),
            int(row["merged_segments"]) - int(row["baseline_merged_segments"]),
        ) if row else None
        if observed != expected:
            raise ValueError(f"CORE+RESCUE result differs from the documented comparison for {variant}: {observed}")

    complementarity_summary = read_json(COMPLEMENTARITY / "summary_v1.json")
    complementarity_scope = complementarity_summary.get("scope", "")
    if not all(value in complementarity_scope for value in ("TRAIN only", "22 matches", "5,372 common candidate groups", "130 completed catalog goals")):
        raise ValueError("Complementarity matrix is not the expected 22-match/130-goal TRAIN population.")
    complementarity_rows = [
        row for row in read_csv(COMPLEMENTARITY / "pairwise_complementarity_v1.csv")
        if int(row["k_per_match"]) == 60
        and row["method_a"] == "fusion_score_goal"
        and row["method_b"] == "net_motion"
    ]
    if len(complementarity_rows) != 1:
        raise ValueError("Expected exactly one K60 fusion/net-motion complementarity row.")
    complementarity = complementarity_rows[0]
    observed_complementarity = (
        int(complementarity["candidate_intersection"]),
        int(complementarity["candidate_union"]),
        round(float(complementarity["candidate_jaccard"]), 6),
        int(complementarity["goal_intersection"]),
        int(complementarity["goal_union_covered"]),
        round(float(complementarity["goal_jaccard"]), 6),
        int(complementarity["goals_only_a"]),
        int(complementarity["goals_only_b"]),
        int(complementarity["union_merged_segments"]),
        round(float(complementarity["union_unique_review_minutes_per_match"]), 3),
    )
    expected_complementarity = (844, 1796, 0.469933, 98, 121, 0.809917, 13, 10, 983, 68.744)
    if observed_complementarity != expected_complementarity:
        raise ValueError(f"K60 fusion/net-motion complementarity changed: {observed_complementarity}")
    method_rows = {
        row["method"]: row
        for row in read_csv(COMPLEMENTARITY / "method_coverage_v1.csv")
        if int(row["k_per_match"]) == 60 and row["method"] in ("fusion_score_goal", "net_motion")
    }
    expected_methods = {
        "fusion_score_goal": (111, 1320, 1028, 58.231),
        "net_motion": (108, 1320, 1037, 58.278),
    }
    if set(method_rows) != set(expected_methods):
        raise ValueError("K60 individual fusion/net-motion method rows are missing from the complementarity matrix.")
    for method, expected in expected_methods.items():
        row = method_rows[method]
        observed = (
            int(row["covered_goals"]),
            int(row["selected_candidates"]),
            int(row["merged_segments"]),
            round(float(row["unique_review_minutes_per_match"]), 3),
        )
        if observed != expected:
            raise ValueError(f"K60 complementarity baseline changed for {method}: {observed}")

    expected_rrf_k60 = {
        "RRF3_temporal_visual_fusion": (113, 58.185),
        "RRF4_plus_local_trajectory": (111, 58.297),
        "RRF4_plus_shot_motion": (113, 58.304),
        "RRF4_plus_net_motion": (114, 58.336),
        "RRF6_plus_all_three": (112, 58.265),
    }
    rrf_rows = [
        row for row in read_csv(ROOT / "experiments/ball/GOAL-EVENT-BALL-EVIDENCE-RRF-ENSEMBLE-EXP-0001/recall_at_k_comparison_v1.csv")
        if int(row["k_per_match"]) == 60 and row["method"] in expected_rrf_k60
    ]
    rrf_k60 = {row["method"]: (int(row["covered_goals"]), round(float(row["unique_review_minutes_per_match"]), 3)) for row in rrf_rows}
    if rrf_k60 != expected_rrf_k60:
        raise ValueError(f"RRF specialist ablation K60 differs from the consolidated report: {rrf_k60}")

    gate = read_json(DETECTOR_GATE).get("gate", {})
    baseline_gate = gate.get("baseline", {})
    oversampled_gate = gate.get("oversampled", {})
    same_holdout = baseline_gate.get("positive_frames") == oversampled_gate.get("positive_frames") == 104
    computed_pass = (
        oversampled_gate.get("candidate_recall", 0) >= gate.get("minimum_oversampled_recall", 1)
        and oversampled_gate.get("candidate_recall", 0) > baseline_gate.get("candidate_recall", 1)
    )
    if (
        not gate.get("passed")
        or gate.get("decision") != "REOPEN_TRACKER"
        or not same_holdout
        or not computed_pass
        or baseline_gate.get("hits") != 71
        or oversampled_gate.get("hits") != 72
    ):
        raise ValueError("Detector-only conditional gate does not match the frozen 71/104 vs. 72/104 result.")

    prior = read_csv(CONTROLS_DIR / "pairwise_control_audit_v2_prior50_annotated.csv")
    batch2 = read_csv(CONTROLS_DIR / "pairwise_control_audit_v2_batch2_annotated.csv")
    controls = prior + batch2
    if len(controls) != 100 or any(
        row.get("manual_status") != "COMPLETED" or row.get("manual_label") != "NO_GOAL"
        for row in controls
    ):
        raise ValueError("Final freeze requires 100 completed human-confirmed NO_GOAL controls.")
    hardneg = read_json(HARDNEG_DIR / "reviewed_hard_negative_reranking.json")
    if (
        hardneg.get("manually_confirmed_no_goal_groups") != 100
        or "TRAIN only" not in hardneg.get("scope", "")
        or "weak_label=0 excluded" not in hardneg.get("scope", "")
        or hardneg.get("baseline_gap45_k60_reproduced", [None, None, None, None])[-1] != 128
    ):
        raise ValueError("Hard-negative checkpoint metadata does not match the frozen 100-control TRAIN protocol.")
    pairwise = {row["method"]: row for row in hardneg.get("pairwise_summary", [])}
    expected_pairwise = {
        "RF_BASELINE_FROZEN": 0,
        "LR_BASE36_HARDNEG_LOMO": 59,
        "LR_RICH49_HARDNEG_LOMO": 64,
        "RF_RICH49_HARDNEG_LOMO": 81,
        "XGB_RICH49_HARDNEG_LOMO": 64,
    }
    if any(
        method not in pairwise
        or pairwise[method].get("matched_pairs") != 100
        or pairwise[method].get("goals_above_reviewed_no_goal") != wins
        for method, wins in expected_pairwise.items()
    ):
        raise ValueError("Hard-negative pairwise results differ from the reviewed 100-pair checkpoint.")
    hardneg_k60 = {
        row["method"]: int(row["covered_catalog_goals"])
        for row in read_csv(HARDNEG_DIR / "reviewed_hard_negative_rerank_summary.csv")
        if int(row["k"]) == 60
    }
    expected_recall_k60 = {
        "RF_BASELINE_FROZEN": 128,
        "LR_BASE36_HARDNEG_LOMO": 76,
        "LR_RICH49_HARDNEG_LOMO": 88,
        "RF_RICH49_HARDNEG_LOMO": 81,
        "XGB_RICH49_HARDNEG_LOMO": 91,
    }
    if hardneg_k60 != expected_recall_k60:
        raise ValueError(f"Hard-negative Recall@60 comparison changed: {hardneg_k60}")

    anchor_primary = {
        (row["run"], row["method"]): row
        for row in read_csv(ANCHOR_SUMMARY)
        if row["scope"] == "PRIMARY_CENTER_16_26"
    }
    expected_anchor_primary = {
        ("full_baseline", "ANCHOR_TRAJECTORY"): (0.037618, 0.015674, 0.032812, 0.774295),
        ("full_baseline", "VITERBI"): (0.084639, 0.047022, 0.021444, 0.780564),
        ("full_oversampled", "ANCHOR_TRAJECTORY"): (0.021944, 0.040752, 0.019356, 0.714734),
        ("full_oversampled", "VITERBI"): (0.112853, 0.065831, 0.018876, 0.742947),
    }
    if set(anchor_primary) != set(expected_anchor_primary):
        raise ValueError("Anchor-trajectory primary table no longer has the four expected detector/method rows.")
    for key, expected in expected_anchor_primary.items():
        row = anchor_primary[key]
        observed = tuple(round(float(row[column]), 6) for column in (
            "identity_accuracy_iou50",
            "distractor_switch_rate_iou50_on_visible",
            "mean_normalized_jump_per_frame",
            "selected_box_coverage_on_visible",
        ))
        if observed != expected:
            raise ValueError(f"Anchor-trajectory primary metrics changed for {key}: {observed}")
    for detector in ("full_baseline", "full_oversampled"):
        anchor = anchor_primary[(detector, "ANCHOR_TRAJECTORY")]
        viterbi = anchor_primary[(detector, "VITERBI")]
        if not (
            float(anchor["identity_accuracy_iou50"]) < float(viterbi["identity_accuracy_iou50"])
            and float(anchor["distractor_switch_rate_iou50_on_visible"]) < float(viterbi["distractor_switch_rate_iou50_on_visible"])
            and float(anchor["mean_normalized_jump_per_frame"]) >= float(viterbi["mean_normalized_jump_per_frame"])
        ):
            raise ValueError(f"Anchor-trajectory NO-GO interpretation no longer follows from {detector} primary metrics.")

    annotation_freeze = read_json(TRACKER_DIR / "annotation_freeze_v1.json")
    if annotation_freeze.get("annotations_sha256") != sha256(ANNOTATIONS):
        raise ValueError("Current BALL/geometry labels differ from the frozen tracker-scoring annotations.")
    if annotation_freeze.get("event_manifest_sha256") != sha256(TRACKER_DIR / "event_manifest_v1.csv"):
        raise ValueError("Current tracker event manifest differs from the annotation freeze.")

    run_freeze = read_json(TRACKER_DIR / "blind_tracker_run_freeze_v1.json")
    if not run_freeze.get("prediction_only"):
        raise ValueError("Final freeze requires the tracker runs to be explicitly label-blind prediction-only.")
    tracker_summary = read_json(TRACKER_DIR / "tracker_retest_metrics_v1.json")
    if tracker_summary.get("annotation_freeze", {}).get("sha256") != sha256(TRACKER_DIR / "annotation_freeze_v1.json"):
        raise ValueError("Spatial tracker summary is not scored from the current annotation freeze.")
    if tracker_summary.get("tracker_configuration") != {
        "confidence": 0.01,
        "max_candidates": 10,
        "imgsz": 960,
        "device": "cpu",
        "verifier": None,
    }:
        raise ValueError("Tracker settings differ from the frozen conditional retest configuration.")
    tracking_aggregate: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in read_csv(TRACKER_DIR / "tracker_retest_group_metrics_v1.csv"):
        key = (row["window"], row["detector"], row["event_role"])
        for metric in ("target_frames", "candidate_hits", "identity_hits"):
            tracking_aggregate[key][metric] += int(row[metric])
    expected_tracking = {
        ("center", "baseline", "CATALOGED_GOAL"): (264, 76, 29),
        ("center", "oversampled", "CATALOGED_GOAL"): (264, 72, 31),
        ("full", "baseline", "CATALOGED_GOAL"): (1009, 397, 219),
        ("full", "oversampled", "CATALOGED_GOAL"): (1009, 414, 216),
    }
    for key, expected in expected_tracking.items():
        observed = tuple(tracking_aggregate[key][metric] for metric in ("target_frames", "candidate_hits", "identity_hits"))
        if observed != expected:
            raise ValueError(f"Viterbi positive tracking aggregate changed for {key}: {observed}")
    spatial_pair_outcomes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pair_rows = read_csv(TRACKER_DIR / "tracker_retest_pair_metrics_v1.csv")
    for row in pair_rows:
        spatial_pair_outcomes[(row["window"], row["detector"])][row["aperture_occupancy_outcome"]] += 1
    expected_spatial_pairs = {
        ("center", "baseline"): Counter({"POSITIVE_HIGHER_APERTURE_OCCUPANCY": 1, "CONTROL_HIGHER_APERTURE_OCCUPANCY": 3, "TIE": 2}),
        ("center", "oversampled"): Counter({"POSITIVE_HIGHER_APERTURE_OCCUPANCY": 2, "CONTROL_HIGHER_APERTURE_OCCUPANCY": 1, "TIE": 3}),
        ("full", "baseline"): Counter({"CONTROL_HIGHER_APERTURE_OCCUPANCY": 3, "NOT_EVALUABLE": 3}),
        ("full", "oversampled"): Counter({"POSITIVE_HIGHER_APERTURE_OCCUPANCY": 1, "TIE": 1, "NOT_EVALUABLE": 4}),
    }
    if len(pair_rows) != 24 or dict(spatial_pair_outcomes) != expected_spatial_pairs:
        raise ValueError(f"Projected geometry pair outcomes differ from the consolidated report: {dict(spatial_pair_outcomes)}")
    volume_fractions: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in read_csv(TRACKER_DIR / "tracker_retest_sequence_metrics_v1.csv"):
        value = row.get("center_inside_volume_fraction", "")
        if value:
            key = (row["window"], row["detector"], row["event_role"])
            volume_fractions[key].append(float(value))
    observed_volume_means = {
        key: (len(values), round(sum(values) / len(values), 4))
        for key, values in volume_fractions.items()
    }
    expected_volume_means = {
        ("center", "baseline", "CATALOGED_GOAL"): (29, 0.1285),
        ("center", "baseline", "CONFIRMED_NO_GOAL"): (6, 0.2424),
        ("center", "oversampled", "CATALOGED_GOAL"): (29, 0.1852),
        ("center", "oversampled", "CONFIRMED_NO_GOAL"): (6, 0.1364),
        ("full", "baseline", "CATALOGED_GOAL"): (23, 0.1502),
        ("full", "baseline", "CONFIRMED_NO_GOAL"): (5, 0.3455),
        ("full", "oversampled", "CATALOGED_GOAL"): (22, 0.1387),
        ("full", "oversampled", "CONFIRMED_NO_GOAL"): (5, 0.0727),
    }
    if observed_volume_means != expected_volume_means:
        raise ValueError(f"Projected-volume means differ from the consolidated report: {observed_volume_means}")
    for detector, checkpoint in tracker_summary.get("detector_checkpoints", {}).items():
        checkpoint_path = ROOT / checkpoint["path"].replace("\\", "/")
        verify_file_hash(checkpoint_path, checkpoint["sha256"], f"{detector} checkpoint")
        frozen_checkpoint = run_freeze.get("detector_checkpoints", {}).get(detector, {}).get("sha256")
        if checkpoint["sha256"] != frozen_checkpoint:
            raise ValueError(f"Tracker run freeze disagrees on the {detector} checkpoint hash.")
    for run_name, run in tracker_summary.get("runs", {}).items():
        frame_path = ROOT / run["frame_results_path"].replace("\\", "/")
        verify_file_hash(frame_path, run["frame_results_sha256"], f"{run_name} frame results")
        run_dir = frame_path.parent
        frozen_run = run_freeze.get("runs", {}).get(run_dir.name)
        if not frozen_run or frozen_run.get("frame_results_sha256") != run["frame_results_sha256"]:
            raise ValueError(f"Tracker run freeze disagrees on {run_name} prediction output.")
        input_path = TRACKER_DIR / frozen_run["input_path"]
        verify_file_hash(input_path, frozen_run["input_sha256"], f"{run_name} blind input")
        trajectory_summary = run_dir / "trajectory_identity_summary.csv"
        verify_file_hash(trajectory_summary, run["trajectory_summary_sha256"], f"{run_name} trajectory summary")
        metrics_path = run_dir / "metrics.json"
        verify_file_hash(metrics_path, frozen_run["metrics_sha256"], f"{run_name} run metadata")
        if not read_json(metrics_path).get("prediction_only"):
            raise ValueError(f"Tracker run {run_name} is not marked label-blind prediction-only.")

    state = interim["validated_state"]
    if not state.get("full_geometry_freeze_ready"):
        raise ValueError("Latest interim snapshot does not confirm full geometry freeze readiness.")
    if interim.get("decisions", {}).get("v4") != "UNCHANGED; NO_PROMOTION_AUTHORIZED":
        raise ValueError("Final freeze must preserve the current V4 no-promotion decision.")
    if interim.get("decisions", {}).get("test") != "CLOSED":
        raise ValueError("Final freeze must preserve the closed TEST decision.")

    if args.preflight:
        print(json.dumps({
            "status": "FINAL_FREEZE_PREFLIGHT_OK",
            "planned_final_freeze_date": PLANNED_DATE.isoformat(),
            "current_date": today.isoformat(),
            "source_interim_manifest": interim_path.relative_to(ROOT).as_posix(),
            "artifact_count": len(artifacts),
            "tracker_runs_hash_verified": len(tracker_summary.get("runs", {})),
            "writes_final_manifest": False,
        }, ensure_ascii=False))
        return 0

    result = {
        "freeze_id": "REPORT87-PD-SPRINT-FINAL-FREEZE-20261011",
        "manifest_version": 1,
        "is_final_sprint_freeze": True,
        "planned_final_freeze_date": PLANNED_DATE.isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "TRAIN-only evidence; TEST closed; V4 unchanged and not promoted",
        "source_interim_manifest": {
            "path": interim_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(interim_path),
            "manifest_version": interim["manifest_version"],
        },
        "validated_state": {
            "ball_annotations": {"completed": len(annotations), "expected": 1435},
            "positive_miss_visual_review": {
                "completed": len(misses),
                "expected": 127,
                "overlapping_tag_counts_verified": len(expected_miss_tags),
                "objective_status_columns_verified": 2,
            },
            "detector_only_gate": {"baseline_hits": gate["baseline"]["hits"], "oversampled_hits": gate["oversampled"]["hits"], "positive_frames": gate["baseline"]["positive_frames"]},
            "core_rescue_key_variants_verified": len(expected_core),
            "complementarity_k60_verified": {
                "candidate_intersection": observed_complementarity[0],
                "candidate_union": observed_complementarity[1],
                "goal_intersection": observed_complementarity[3],
                "goal_union_covered": observed_complementarity[4],
                "fusion_only_goals": observed_complementarity[6],
                "net_motion_only_goals": observed_complementarity[7],
            },
            "hard_negative_pairwise_results_verified": len(expected_pairwise),
            "hard_negative_recall_at_60_verified": hardneg_k60,
            "anchor_primary_rows_verified": len(expected_anchor_primary),
            "human_confirmed_no_goal_controls": {"completed": len(controls), "expected": 100},
            "goal_geometry_event_centers": {"completed_valid_region_and_volume": len(centers), "expected": 35},
            "tracker_prediction_runs_hash_verified": len(tracker_summary.get("runs", {})),
            "viterbi_positive_aggregates_verified": len(expected_tracking),
            "spatial_pair_outcomes_verified": len(pair_rows),
            "projected_volume_means_verified": len(observed_volume_means),
            "interim_artifacts_hash_verified": len(artifacts),
        },
        "decisions": interim["decisions"],
        "artifacts": artifacts,
    }
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FINAL_FREEZE_WRITTEN", "path": str(output), "artifacts": len(artifacts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
