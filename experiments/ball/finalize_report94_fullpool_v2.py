"""Valida e congela os modos full-pool V2 antes de qualquer acesso a TEST."""
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
INTERIM_DIR = ROOT / "experiments/ball/REPORT87-PD-SPRINT-INTERIM-FREEZE-20261005"
CANONICAL_FINAL_DIR = ROOT / "experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011"
OUTPUT = CANONICAL_FINAL_DIR / "final_freeze_manifest_v2.json"
NESTED_DIR = ROOT / "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1"
SUMMARY_DIR = ROOT / "experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1"
MODES_DIR = ROOT / "experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2"


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


def latest_interim() -> Path:
    paths = sorted(INTERIM_DIR.glob("interim_freeze_manifest_v*.json"), key=lambda p: int(p.stem.rsplit("v", 1)[1]))
    if not paths:
        raise FileNotFoundError("Não há snapshot intermediário para validar.")
    return paths[-1]


def check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Artefato ausente ({label}): {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"Hash divergente em {label}: esperado {expected}, encontrado {observed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Validar sem gravar o manifesto final.")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.preflight:
        raise FileExistsError(f"Freeze V2 já existe; não será sobrescrito: {OUTPUT}")

    preflight = subprocess.run(
        [sys.executable, str(ROOT / "experiments/ball/finalize_report87_sprint_v1.py"), "--preflight"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    legacy_check = json.loads(preflight.stdout)
    if legacy_check.get("status") != "FINAL_FREEZE_PREFLIGHT_OK":
        raise ValueError("O preflight histórico não aprovou as evidências anteriores do sprint.")

    interim_path = latest_interim()
    interim = read_json(interim_path)
    if interim.get("is_final_sprint_freeze") is not False:
        raise ValueError("O snapshot de entrada não está marcado como intermediário.")
    if interim.get("decisions", {}).get("test") != "CLOSED":
        raise ValueError("O snapshot de entrada não mantém TEST fechado.")
    for rel_path, metadata in interim.get("artifacts", {}).items():
        check_hash(ROOT / rel_path, metadata["sha256"], rel_path)

    nested_meta = read_json(NESTED_DIR / "summary_v1.json")
    if (nested_meta.get("matches"), nested_meta.get("catalog_goals"), nested_meta.get("common_candidate_groups"), nested_meta.get("outer_folds")) != (25, 152, 5742, 25):
        raise ValueError("As dimensões declaradas do Nested full-pool não são as esperadas.")
    folds = read_csv(NESTED_DIR / "nested_time_budget_outer_folds_v1.csv")
    goals = read_csv(NESTED_DIR / "nested_time_budget_goal_coverage_v1.csv")
    summary = read_csv(NESTED_DIR / "nested_time_budget_summary_v1.csv")
    if len(folds) != 375 or len(goals) != 2280 or len(summary) != 15:
        raise ValueError("Dimensões inesperadas nos resultados Nested full-pool.")
    outer_keys = {(r["budget_minutes_per_match"], r["policy_family"], r["outer_match"]) for r in folds}
    goal_keys = {(r["budget_minutes_per_match"], r["policy_family"], r["match_id"], r["goal_catalog_id"]) for r in goals}
    if len(outer_keys) != 375 or len(goal_keys) != 2280:
        raise ValueError("Foram encontradas linhas duplicadas nos resultados Nested.")
    if {r["outer_match"] for r in folds}.__len__() != 25:
        raise ValueError("Os folds não cobrem 25 partidas distintas.")
    if any(r["outer_budget_exceeded"].strip().lower() in {"true", "1", "yes"} for r in folds):
        raise ValueError("Um ou mais folds externos excederam o orçamento congelado.")
    for budget in {r["budget_minutes_per_match"] for r in folds}:
        for family in {r["policy_family"] for r in folds}:
            rows = [r for r in goals if r["budget_minutes_per_match"] == budget and r["policy_family"] == family]
            if len(rows) != 152 or len({r["match_id"] for r in rows}) != 25 or len({r["goal_catalog_id"] for r in rows}) != 152:
                raise ValueError(f"Cobertura incompleta em budget={budget}, family={family}.")

    summary_manifest = read_json(SUMMARY_DIR / "summary_v1.json")
    if summary_manifest.get("bootstrap", {}).get("replicates") != 10000 or summary_manifest.get("bootstrap", {}).get("seed") != 901006:
        raise ValueError("Parâmetros do bootstrap não correspondem ao protocolo registrado.")
    if len(read_csv(SUMMARY_DIR / "nested_mode_per_match_oof_v1.csv")) != 75:
        raise ValueError("A tabela OOF por partida não contém 75 linhas.")
    if len(read_csv(SUMMARY_DIR / "nested_mode_goal_coverage_oof_v1.csv")) != 456:
        raise ValueError("A tabela OOF por gol não contém 456 linhas.")
    if len(read_csv(SUMMARY_DIR / "nested_pareto_points_oof_v1.csv")) != 15:
        raise ValueError("A fronteira Pareto não contém os 15 pontos Nested.")

    mode_config = read_json(MODES_DIR / "final_modes_v2.json")
    run_manifest = read_json(MODES_DIR / "run_manifest_v2.json")
    if mode_config.get("status") != "TRAIN_SELECTED_CANDIDATES_NOT_YET_FINAL_FREEZE":
        raise ValueError("Status inesperado nas configurações V2.")
    if run_manifest.get("validation_test_policy") != "neither VALIDATION nor TEST was opened or used":
        raise ValueError("O manifesto de seleção não garante isolamento de TEST/VALIDATION.")
    expected_modes = {
        "FAST_V2": (30, "NO_RESCUE", "TEMPORAL", 45, 40),
        "BALANCED_V2": (40, "CORE_RESCUE", "CORE40_RRF4_NET_RESCUE", 60, 40),
        "HIGH_RECALL_V2": (58, "CORE_RESCUE", "CORE40_RRF4_NET_RESCUE", 60, 40),
    }
    observed_modes = mode_config.get("modes", {})
    if set(observed_modes) != set(expected_modes):
        raise ValueError("O conjunto de modos V2 não corresponde ao protocolo congelado.")
    for name, expected in expected_modes.items():
        mode = observed_modes[name]
        observed = (mode["budget_minutes_per_match"], mode["selection_family"], mode["method"], mode["window_seconds"], mode["k_per_match"])
        if observed != expected:
            raise ValueError(f"Configuração inesperada para {name}: {observed}")
    mode_selection_rows = read_csv(MODES_DIR / "final_mode_selection_v2.csv")
    if len(mode_selection_rows) != 3 or any(
        row.get("metric_role") != "in_sample_selection_only_not_generalization_estimate"
        for row in mode_selection_rows
    ):
        raise ValueError("A seleção fixa não identifica as métricas como in-sample.")

    model_manifest = read_json(MODES_DIR / "model_manifest_v2.json")
    for model_name, metadata in model_manifest.items():
        model_path = ROOT / metadata["path"]
        check_hash(model_path, metadata["sha256"], f"modelo {model_name}")
    selection_inputs = {
        "temporal_features": "results/local_trajectory_dataset_train_fullpool_v1/temporal_group_dataset_local_trajectory_v1.csv",
        "net_features": "results/net_motion_dataset_train_fullpool_nested_v1/temporal_group_dataset_net_motion_v1.csv",
        "candidate_scores": "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv",
        "candidate_embeddings": "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.npy",
        "candidate_embedding_manifest": "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.json",
        "verifier_predictions": "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/predictions_lomo_v1.csv",
        "verifier_embeddings": "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/sequence_embeddings_v1.npz",
        "goal_catalog": "results/local_trajectory_dataset_train_fullpool_v1/goal_timestamps_train_subset_v1.csv",
        "frozen_temporal_model": "experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib",
    }
    if set(run_manifest.get("inputs_sha256", {})) != set(selection_inputs):
        raise ValueError("A lista de entradas da seleção V2 diverge do protocolo registrado.")
    for input_name, relative_path in selection_inputs.items():
        check_hash(ROOT / relative_path, run_manifest["inputs_sha256"][input_name], f"entrada de seleção {input_name}")

    if args.preflight:
        print(json.dumps({
            "status": "REPORT94_FULLPOOL_FREEZE_PREFLIGHT_OK",
            "source_interim_manifest": interim_path.relative_to(ROOT).as_posix(),
            "source_interim_sha256": sha256(interim_path),
            "legacy_preflight": legacy_check,
            "nested": {"matches": 25, "candidate_groups": 5742, "catalog_goals": 152, "outer_rows": len(folds)},
            "bootstrap_replicates": 10000,
            "modes": expected_modes,
            "test_accessed": False,
            "writes_final_manifest": False,
        }, ensure_ascii=False))
        return 0

    CANONICAL_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "freeze_id": "REPORT87-PD-SPRINT-FINAL-FREEZE-20261011",
        "manifest_version": 2,
        "is_final_sprint_freeze": True,
        "supersedes": "final_freeze_manifest_v1.json",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "TRAIN-only evidence; TEST closed; V4 unchanged and not promoted",
        "source_interim_manifest": {
            "path": interim_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(interim_path),
            "manifest_version": interim["manifest_version"],
        },
        "legacy_preflight": legacy_check,
        "validated_state": {
            "fullpool_nested": {"matches": 25, "candidate_groups": 5742, "catalog_goals": 152, "outer_folds": 25, "outer_rows": len(folds), "goal_coverage_rows": len(goals), "budget_violations": 0},
            "paired_bootstrap": {"replicates": 10000, "seed": 901006, "resampling_unit": "match_id", "paired": True},
            "pareto_points": 15,
            "fixed_mode_configs_verified": 3,
            "model_hashes_verified": len(model_manifest),
            "test_accessed_before_freeze": False,
        },
        "decisions": {
            "test": "CLOSED_UNTIL_THIS_V2_FREEZE_IS_WRITTEN",
            "modes": "FAST_V2, BALANCED_V2 and HIGH_RECALL_V2 frozen for one TEST evaluation; no retuning or winner selection afterward",
            "v4": "UNCHANGED; NO_PROMOTION_AUTHORIZED",
        },
        "frozen_test_protocol": {
            "modes": {name: observed_modes[name] for name in sorted(observed_modes)},
            "metrics": ["catalog-goal recall", "unique video minutes per match", "merged segments per match", "selected candidates per match", "computational analysis time", "paired per-match gains/losses"],
            "goal_denominator": "only catalog timestamps marked completed; NOT_EVALUABLE records are excluded and counted separately",
            "aggregation": "report each mode overall and per match; keep all three modes, do not select a winner using TEST",
            "minutes": "union duration of selected merged segments per match",
            "cancellation_or_retuning": "no retuning, grid search or second TEST run after results are observed",
        },
        "artifacts": {
            rel_path: {"sha256": metadata["sha256"], "bytes": metadata["bytes"]}
            for rel_path, metadata in interim["artifacts"].items()
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORT94_FULLPOOL_FINAL_FREEZE_WRITTEN", "path": OUTPUT.relative_to(ROOT).as_posix(), "test_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
