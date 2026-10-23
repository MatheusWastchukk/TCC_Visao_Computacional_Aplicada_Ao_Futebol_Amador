"""Materialize the frozen-grid mode selection and reuse the V2 TRAIN model bundle."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3"
V2 = ROOT / "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    selection_path = OUT / "final_mode_selection_v3.csv"
    selection = read_csv(selection_path)
    if not selection:
        raise ValueError("LOMO selector emitted no operating modes")
    model_manifest = json.loads((V2 / "model_manifest_v2.json").read_text(encoding="utf-8"))
    for metadata in model_manifest.values():
        path = ROOT / metadata["path"]
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"V2 TRAIN model hash changed: {path}")
    (OUT / "model_manifest_v3.json").write_text(json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    modes: dict[str, dict[str, object]] = {}
    for row in selection:
        mode_name = row["mode"]
        method = row["method"]
        sources = {
            "TEMPORAL": ["temporal_ranker"],
            "RRF3": ["temporal_ranker", "visual_verifier", "fusion_ranker"],
            "RRF4_NET": ["temporal_ranker", "visual_verifier", "fusion_ranker", "net_motion_ranker"],
        }[method]
        modes[mode_name] = {
            "budget_minutes_per_match": int(row["budget_minutes_per_match"]),
            "selection_family": "NO_RESCUE",
            "method": method,
            "window_seconds": int(row["window_seconds"]),
            "k_per_match": int(row["k_per_match"]),
            "gap_seconds": float(row["gap_seconds"]),
            "lomo_covered_goals_for_selection": int(row["lomo_covered_goals_for_selection"]),
            "lomo_catalog_goals": int(row["lomo_catalog_goals"]),
            "lomo_selection_recall": row["lomo_selection_recall"],
            "lomo_mean_unique_minutes": row["lomo_mean_unique_minutes"],
            "lomo_max_match_unique_minutes": row["lomo_max_match_unique_minutes"],
            "metric_role": row["metric_role"],
            "models_manifest": "model_manifest_v3.json",
            "rescue_enabled": False,
            "ranking_sources": sources,
        }
    config = {
        "config_id": "REPORT98-NO-RESCUE-MODES-LOMO-V3",
        "status": "TRAIN_LOMO_CV_SELECTED_CANDIDATES_NOT_YET_FINAL_FREEZE",
        "scope": "Full 25-match TRAIN only; VALIDATION and TEST not accessed",
        "selection_metric_warning": "The fixed grid was selected using the same 25 outer LOMO outcomes; selected coverage is configuration-selection evidence, not an independent performance estimate. Nested 25/25 remains the primary estimate of the budget-selection procedure.",
        "budgets_are": "unique selected video minutes per match; not human review time",
        "grid": {
            "methods": ["TEMPORAL", "RRF3", "RRF4_NET"],
            "budgets": [30, 40, 58],
            "windows_seconds": [30, 40, 45, 60],
            "k_per_match": [20, 30, 40, 50, 55, 60],
            "gap_seconds": 45.0,
        },
        "modes": modes,
        "mode_results_file": "final_mode_selection_v3.csv",
        "lomo_grid_file": "fixed_policy_lomo_summary_v3.csv",
        "oof_rows_file": "fixed_policy_lomo_by_match_v3.csv",
        "models": "Reuse unchanged full-TRAIN model bundle from REPORT94 V2; only no-rescue policies are enabled.",
    }
    (OUT / "final_modes_v3.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "experiment_id": "REPORT98-NO-RESCUE-MODES-LOMO-V3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_is_configuration_selection": True,
        "generalization_estimates_source": "NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1",
        "matches": 25,
        "candidate_groups": 5742,
        "catalog_goals": 152,
        "selected_modes": list(modes),
        "parent_model_bundle": "REPORT94-FULLPOOL-FINAL-MODES-V2",
        "model_hashes": model_manifest,
        "selection_manifest_sha256": sha256(OUT / "selection_manifest_v3.json"),
        "selection_table_sha256": sha256(selection_path),
        "validation_test_policy": "neither VALIDATION nor TEST was opened or used",
    }
    (OUT / "run_manifest_v3.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"modes": list(modes), "output": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
