"""Validate TRAIN LOMO mode selection, then freeze V3 while TEST remains closed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3"
FINAL_DIR = ROOT / "experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011"
FINAL_V3 = FINAL_DIR / "final_freeze_manifest_v3.json"
INTERIM_DIR = ROOT / "experiments/ball/REPORT87-PD-SPRINT-INTERIM-FREEZE-20261005"
NESTED = ROOT / "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if FINAL_V3.exists() and not args.preflight:
        raise FileExistsError(f"Freeze V3 already exists; refusing overwrite: {FINAL_V3}")

    v2_preflight = subprocess.run(
        [sys.executable, str(ROOT / "experiments/ball/finalize_report94_fullpool_v2.py"), "--preflight"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    if json.loads(v2_preflight.stdout).get("status") != "REPORT94_FULLPOOL_FREEZE_PREFLIGHT_OK":
        raise ValueError("V2 historical evidence preflight did not pass")
    interim_paths = sorted(INTERIM_DIR.glob("interim_freeze_manifest_v*.json"), key=lambda p: int(p.stem.rsplit("v", 1)[1]))
    if not interim_paths:
        raise FileNotFoundError("No interim snapshot exists")
    interim_path = interim_paths[-1]
    interim = read_json(interim_path)
    if interim.get("is_final_sprint_freeze") is not False or interim.get("decisions", {}).get("test") != "CLOSED":
        raise ValueError("Latest interim snapshot does not keep TEST closed")
    for rel_path, metadata in interim.get("artifacts", {}).items():
        observed = sha256(ROOT / rel_path)
        if observed != metadata["sha256"]:
            raise ValueError(f"Interim artifact hash mismatch: {rel_path}")

    nested = read_json(NESTED / "summary_v1.json")
    if (nested.get("matches"), nested.get("catalog_goals"), nested.get("common_candidate_groups"), nested.get("outer_folds")) != (25, 152, 5742, 25):
        raise ValueError("Primary Nested 25/25 result dimensions changed")
    selection = read_csv(OUT / "final_mode_selection_v3.csv")
    grid = read_csv(OUT / "fixed_policy_lomo_summary_v3.csv")
    oof = read_csv(OUT / "fixed_policy_lomo_by_match_v3.csv")
    if len(grid) != 72 or len(oof) != 1800:
        raise ValueError(f"Expected 72 grid rows and 1,800 OOF rows; got {len(grid)} and {len(oof)}")
    config_keys = {(r["method"], r["window_seconds"], r["k_per_match"]) for r in grid}
    if len(config_keys) != 72 or any(int(r["matches"]) != 25 or int(r["catalog_goals"]) != 152 for r in grid):
        raise ValueError("LOMO grid dimensions or aggregation denominators are invalid")
    oof_keys = {(r["match_id"], r["method"], r["window_seconds"], r["k_per_match"]) for r in oof}
    if len(oof_keys) != 1800 or {r["match_id"] for r in oof}.__len__() != 25:
        raise ValueError("LOMO rows contain duplicates or do not cover all held matches")
    for row in grid:
        matching = [r for r in oof if (r["method"], r["window_seconds"], r["k_per_match"]) == (row["method"], row["window_seconds"], row["k_per_match"])]
        if len(matching) != 25 or sum(int(r["catalog_goals"]) for r in matching) != 152:
            raise ValueError("An LOMO configuration does not cover all 25 matches and 152 goals")
        if abs(sum(int(r["covered_goals"]) for r in matching) - int(row["covered_goals"])) > 0:
            raise ValueError("LOMO summary coverage does not reconcile to per-match rows")
        if max(float(r["unique_video_minutes"]) for r in matching) > float(row["max_match_unique_video_minutes"]) + 1e-6:
            raise ValueError("LOMO summary max minutes do not reconcile")
    if not selection or len({(r["method"], r["window_seconds"], r["k_per_match"]) for r in selection}) != len(selection):
        raise ValueError("V3 modes are empty or duplicate policies")
    previous_covered = -1
    previous_budget = 0
    for row in selection:
        if row["method"] not in {"TEMPORAL", "RRF3", "RRF4_NET"} or float(row["lomo_max_match_unique_minutes"]) > int(row["budget_minutes_per_match"]) + 1e-6:
            raise ValueError(f"Invalid or over-budget V3 mode: {row}")
        if int(row["budget_minutes_per_match"]) <= previous_budget or int(row["lomo_covered_goals_for_selection"]) <= previous_covered:
            raise ValueError("V3 mode ladder does not increase budget and selected coverage")
        previous_budget = int(row["budget_minutes_per_match"])
        previous_covered = int(row["lomo_covered_goals_for_selection"])

    mode_config = read_json(OUT / "final_modes_v3.json")
    model_manifest = read_json(OUT / "model_manifest_v3.json")
    run_manifest = read_json(OUT / "run_manifest_v3.json")
    if set(mode_config.get("modes", {})) != {r["mode"] for r in selection}:
        raise ValueError("Materialized mode config differs from the LOMO selection table")
    if run_manifest.get("validation_test_policy") != "neither VALIDATION nor TEST was opened or used":
        raise ValueError("V3 run manifest does not preserve TEST isolation")
    for model_name, metadata in model_manifest.items():
        if sha256(ROOT / metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"Model hash mismatch: {model_name}")

    if args.preflight:
        print(json.dumps({"status": "REPORT98_V3_FREEZE_PREFLIGHT_OK", "source_interim_manifest": interim_path.relative_to(ROOT).as_posix(), "modes": len(selection), "grid_configs": len(grid), "oof_rows": len(oof), "nested_primary_unchanged": True, "test_accessed": False, "writes_final_manifest": False}, ensure_ascii=False))
        return 0

    result = {
        "freeze_id": "REPORT87-PD-SPRINT-FINAL-FREEZE-20261011",
        "manifest_version": 3,
        "is_final_sprint_freeze": True,
        "supersedes": "final_freeze_manifest_v2.json",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "TRAIN-only mode selection; TEST closed; Nested 25/25 preserved as primary evidence",
        "source_interim_manifest": {"path": interim_path.relative_to(ROOT).as_posix(), "sha256": sha256(interim_path), "manifest_version": interim["manifest_version"]},
        "historical_v2_preflight": json.loads(v2_preflight.stdout),
        "validated_state": {
            "nested_primary": {"matches": 25, "candidate_groups": 5742, "catalog_goals": 152, "outer_folds": 25},
            "no_rescue_lomo_grid": {"configurations": 72, "oof_rows": 1800, "matches": 25, "catalog_goals": 152, "role": "configuration_selection_only_not_independent_performance_estimate"},
            "selected_mode_count": len(selection),
            "selected_modes": {row["mode"]: {key: row[key] for key in ("budget_minutes_per_match", "method", "window_seconds", "k_per_match", "lomo_covered_goals_for_selection", "lomo_max_match_unique_minutes")} for row in selection},
            "model_hashes_verified": len(model_manifest),
            "test_accessed_before_freeze": False,
        },
        "decisions": {
            "test": "CLOSED_UNTIL_V3_MODE_CATALOG_TIMESTAMPS_ARE_BLINDLY_ANNOTATED",
            "modes": "Freeze only distinct NO_RESCUE modes supported by TRAIN LOMO configuration selection; preserve Nested 25/25 as primary estimate; no tuning or winner selection after TEST",
            "v4": "UNCHANGED; NO_PROMOTION_AUTHORIZED",
        },
        "frozen_test_protocol": {
            "modes": mode_config["modes"],
            "metrics": ["catalog-goal recall", "unique video minutes per match", "merged segments per match", "selected candidates per match", "computational analysis time", "paired per-match gains/losses"],
            "goal_denominator": "only blindly annotated TEST catalog timestamps marked completed; NOT_EVALUABLE records are excluded and counted separately",
            "aggregation": "report each frozen mode overall and per match; keep all frozen modes, do not select a winner using TEST",
            "retuning": "no retuning, grid search or repeat TEST run after results are observed",
        },
        "artifacts": {rel_path: {"sha256": metadata["sha256"], "bytes": metadata["bytes"]} for rel_path, metadata in interim["artifacts"].items()},
    }
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_V3.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORT98_V3_FINAL_FREEZE_WRITTEN", "path": FINAL_V3.relative_to(ROOT).as_posix(), "test_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
