"""Calcula métricas TEST somente após a inferência V3 e o freeze humano."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODES = ("FAST_V3", "BALANCED_V3", "HIGH_RECALL_V3")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Valor booleano invÃ¡lido no manifesto de inferÃªncia: {value!r}")


def interval_union(rows: list[dict[str, str]], duration: float) -> list[tuple[float, float]]:
    intervals = sorted((max(0.0, float(r["delivery_start_seconds"])), min(duration, float(r["delivery_end_seconds"]))) for r in rows)
    result: list[list[float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return [(start, end) for start, end in result]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-freeze", type=Path, default=ROOT / "experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/test_catalog_freeze_manifest_v1.json")
    args = parser.parse_args()
    inference = (ROOT / args.inference).resolve() if not args.inference.is_absolute() else args.inference.resolve()
    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"A saída de métricas não está vazia: {output}")
    inference_manifest_path = inference / "inference_manifest_test_v1.json"
    if not inference_manifest_path.is_file():
        raise FileNotFoundError(f"Inferência ainda não concluída: {inference_manifest_path}")
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    if inference_manifest.get("status") != "TEST_V3_INFERENCE_COMPLETE" or inference_manifest.get("catalog_loaded_by_inference") is not False:
        raise RuntimeError("Manifesto de inferência inválido ou que não garante separação do catálogo.")
    freeze = json.loads(args.catalog_freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "HUMAN_TEST_CATALOG_FROZEN":
        raise RuntimeError("Catálogo humano TEST não está congelado.")
    catalog_path = ROOT / freeze["catalog_path"]
    if sha256(catalog_path) != freeze["catalog_sha256"]:
        raise RuntimeError("Hash do catálogo humano congelado diverge do manifesto.")
    goals, goal_fields = read_csv(catalog_path)
    if len(goals) != 57 or any(row["split"] != "test" or row["annotation_status"] != "completed" for row in goals):
        raise RuntimeError("Esperados 57 registros TEST completed no catálogo congelado.")
    if len({row["goal_catalog_id"] for row in goals}) != len(goals):
        raise RuntimeError("IDs duplicados no catálogo humano congelado.")
    scored, scored_fields = read_csv(inference / "ranked_candidates_test_v1.csv")
    selected, selected_fields = read_csv(inference / "selected_candidates_test_v1.csv")
    inferred_matches, _ = read_csv(inference / "per_match_inference_test_v1.csv")
    match_ids = {row["match_id"] for row in inferred_matches}
    if len(match_ids) != 9 or {row["match_id"] for row in goals} != match_ids:
        raise RuntimeError("As nove partidas do catálogo e da inferência não coincidem.")
    if {row["mode"] for row in selected} != set(MODES):
        raise RuntimeError("A seleção não contém exatamente as três políticas V3.")
    if {row["match_id"] for row in scored} != match_ids:
        raise RuntimeError("O pool ranqueado não cobre as nove partidas.")
    durations = {row["match_id"]: float(row["duration_seconds"]) for row in inferred_matches}
    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in goals:
        goals_by_match[row["match_id"]].append(row)
    selected_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        selected_by_key[(row["match_id"], row["mode"])].append(row)

    per_goal: list[dict[str, Any]] = []
    covered_ids: dict[str, set[str]] = {mode: set() for mode in MODES}
    for goal in goals:
        timestamp = float(goal["event_timestamp_seconds"])
        row: dict[str, Any] = {
            "goal_catalog_id": goal["goal_catalog_id"], "match_id": goal["match_id"],
            "event_timestamp_seconds": f"{timestamp:.3f}", "annotation_status": goal["annotation_status"],
        }
        for mode in MODES:
            candidates = selected_by_key[(goal["match_id"], mode)]
            covering = [c for c in candidates if float(c["delivery_start_seconds"]) <= timestamp <= min(durations[goal["match_id"]], float(c["delivery_end_seconds"]))]
            row[f"{mode}_covered"] = bool(covering)
            row[f"{mode}_covering_candidate_ids"] = ";".join(c["candidate_id"] for c in covering)
            if covering:
                covered_ids[mode].add(goal["goal_catalog_id"])
        per_goal.append(row)
    per_goal_fields = ["goal_catalog_id", "match_id", "event_timestamp_seconds", "annotation_status"] + [f"{mode}_{field}" for mode in MODES for field in ("covered", "covering_candidate_ids")]

    per_match: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {}
    for inference_row in inferred_matches:
        match = inference_row["match_id"]
        match_goals = goals_by_match[match]
        match_row: dict[str, Any] = {
            "match_id": match, "video_filename": inference_row["video_filename"],
            "duration_seconds": float(inference_row["duration_seconds"]), "catalog_goals": len(match_goals),
            "candidate_groups": int(inference_row["candidate_groups"]),
            "miner_elapsed_seconds": inference_row.get("miner_elapsed_seconds", ""),
        }
        for mode in MODES:
            hits = sum(goal["goal_catalog_id"] in covered_ids[mode] for goal in match_goals)
            intervals = interval_union(selected_by_key[(match, mode)], durations[match])
            minutes = sum(end - start for start, end in intervals) / 60.0
            denominator = durations[match]
            prefix = f"{mode}_"
            stored = float(inference_row[f"{mode}_unique_minutes"])
            if abs(stored - minutes) > 1e-5:
                raise RuntimeError(f"União de intervalos diverge da inferência em {match}/{mode}: {stored} vs {minutes}")
            match_row.update({
                prefix + "covered_goals": hits, prefix + "catalog_goals": len(match_goals),
                prefix + "coverage_percent": 100.0 * hits / len(match_goals) if match_goals else None,
                prefix + "missed_goal_ids": ";".join(g["goal_catalog_id"] for g in match_goals if g["goal_catalog_id"] not in covered_ids[mode]),
                prefix + "unique_minutes": minutes, prefix + "video_percent": 100.0 * minutes * 60.0 / denominator,
                prefix + "merged_segments": len(intervals),
                prefix + "selected_candidates": len(selected_by_key[(match, mode)]),
                prefix + "budget_exceeded": parse_bool(inference_row[f"{mode}_budget_exceeded"]),
            })
        per_match.append(match_row)

    for mode in MODES:
        goal_count = sum(len(v) for v in goals_by_match.values())
        covered_count = len(covered_ids[mode])
        keys = [f"{mode}_{field}" for field in ("unique_minutes", "video_percent", "merged_segments", "selected_candidates")]
        aggregate[mode] = {
            "covered_goals": covered_count, "catalog_goals": goal_count,
            "coverage_percent": 100.0 * covered_count / goal_count,
            "mean_unique_minutes_per_match": sum(float(r[f"{mode}_unique_minutes"]) for r in per_match) / len(per_match),
            "mean_video_percent_per_match": sum(float(r[f"{mode}_video_percent"]) for r in per_match) / len(per_match),
            "mean_merged_segments_per_match": sum(int(r[f"{mode}_merged_segments"]) for r in per_match) / len(per_match),
            "mean_selected_candidates_per_match": sum(int(r[f"{mode}_selected_candidates"]) for r in per_match) / len(per_match),
            "total_unique_video_minutes": sum(float(r[f"{mode}_unique_minutes"]) for r in per_match),
            "total_original_video_minutes": sum(float(r["duration_seconds"]) for r in per_match) / 60.0,
            "overall_video_percent": 100.0 * sum(float(r[f"{mode}_unique_minutes"]) for r in per_match) / (sum(float(r["duration_seconds"]) for r in per_match) / 60.0),
            "matches_over_budget": sum(bool(r[f"{mode}_budget_exceeded"]) for r in per_match),
        }
    pairs = []
    for left, right in ((MODES[0], MODES[1]), (MODES[0], MODES[2]), (MODES[1], MODES[2])):
        l_ids, r_ids = covered_ids[left], covered_ids[right]
        pairs.append({
            "mode_a": left, "mode_b": right,
            "covered_by_both": len(l_ids & r_ids), "only_a": len(l_ids - r_ids), "only_b": len(r_ids - l_ids),
            "neither": len(set(g["goal_catalog_id"] for g in goals) - (l_ids | r_ids)),
        })
    output.mkdir(parents=True, exist_ok=True)
    per_match_fields = list(per_match[0])
    write_csv(output / "goal_coverage_test_v1.csv", per_goal, per_goal_fields)
    write_csv(output / "per_match_test_v1.csv", per_match, per_match_fields)
    write_csv(output / "paired_mode_gains_test_v1.csv", pairs, list(pairs[0]))
    summary = {
        "status": "TEST_V3_METRICS_COMPLETE", "scope": "TEST9, 57 timestamps humanos congelados, cobertura de candidatos; não precisão de gols",
        "catalog_freeze_manifest": str(args.catalog_freeze), "catalog_sha256": freeze["catalog_sha256"],
        "inference_manifest": str(inference_manifest_path), "inference_manifest_sha256": sha256(inference_manifest_path),
        "catalog_goals": len(goals), "matches": len(match_ids), "policies": aggregate,
        "paired_gains_losses": pairs,
        "interval_rule": "início <= timestamp humano <= fim, tolerância adicional zero; união de intervalos limitada à duração do vídeo",
        "candidate_pool_groups": len(scored),
        "computational_time": inference_manifest.get("stage_runtime_seconds", {}),
        "warning": "Resultados TEST são descritivos do conjunto independente; não escolher vencedor, ajustar políticas ou retreinar com base neles.",
    }
    (output / "summary_test_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "catalog_goals": len(goals), "policies": aggregate, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
