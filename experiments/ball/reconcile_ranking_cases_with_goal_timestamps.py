"""Concilia janelas candidatas da auditoria com os timestamps catalogados TRAIN."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
CASES = EXPERIMENT / "manual_case_audit.csv"
GOALS = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/train_goal_pool_and_shortlist_coverage_v1.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    cases, goals = read_csv(CASES), read_csv(GOALS)
    case_manifest = read_csv(EXPERIMENT / "case_manifest.csv")
    selection_trace = read_csv(EXPERIMENT / "selection_trace.csv")
    trace_by_group = {row["group_uid"]: row for row in selection_trace}
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in goals:
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds"):
            by_match[goal["match_id"]].append(goal)

    reconciled: list[dict[str, str]] = []
    matched_catalog_ids: set[str] = set()
    role_counts: Counter[str] = Counter()
    role_hit_counts: Counter[str] = Counter()
    for case in cases:
        start = float(case["delivery_start_seconds"])
        end = float(case["delivery_end_seconds"])
        timestamp = float(case["best_timestamp_seconds"])
        matches = [g for g in by_match[case["match_id"]]
                   if start <= float(g["event_timestamp_seconds"]) <= end]
        for role in case["review_role"].split(";"):
            role_counts[role] += 1
            if matches:
                role_hit_counts[role] += 1
        matched_catalog_ids.update(g["goal_catalog_id"] for g in matches)
        reconciled.append({
            "case_id": case["case_id"], "match_id": case["match_id"],
            "group_uid": case["group_uid"], "rank_full_pool": case["temporal_rank_full_pool"],
            "top60_rank": case["top60_rank"], "candidate_timestamp_seconds": f"{timestamp:.3f}",
            "window_start_seconds": f"{start:.3f}", "window_end_seconds": f"{end:.3f}",
            "review_role": case["review_role"],
            "catalog_match_status": "MATCHES_CATALOG_GOAL" if matches else "NO_CATALOG_GOAL_IN_WINDOW",
            "matched_goal_catalog_ids": ";".join(g["goal_catalog_id"] for g in matches),
            "matched_goal_timestamps_seconds": ";".join(g["event_timestamp_seconds"] for g in matches),
            "candidate_minus_goal_seconds": ";".join(f"{timestamp-float(g['event_timestamp_seconds']):.3f}" for g in matches),
            "negative_label_status": "NOT_INFERRED_FROM_CATALOG_ABSENCE",
        })

    output = EXPERIMENT / "timestamp_reconciliation.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reconciled[0]))
        writer.writeheader()
        writer.writerows(reconciled)

    missed_goals = [g for g in case_manifest
                    if g["pool_recoverable"] == "True" and g["covered_by_existing_top60"] == "False"]
    miss_diagnostics: list[dict[str, str]] = []
    for goal in missed_goals:
        trace = trace_by_group[goal["best_temporal_group_uid"]]
        suppressor = trace.get("gap_suppressor_group_uid", "")
        suppressor_trace = trace_by_group.get(suppressor, {})
        miss_diagnostics.append({
            "goal_catalog_id": goal["goal_catalog_id"], "match_id": goal["match_id"],
            "goal_timestamp_seconds": goal["goal_timestamp_seconds"],
            "best_group_uid": goal["best_temporal_group_uid"],
            "best_group_timestamp_seconds": trace["best_timestamp_seconds"],
            "candidate_minus_goal_seconds": f"{float(trace['best_timestamp_seconds'])-float(goal['goal_timestamp_seconds']):.3f}",
            "best_temporal_rank_full_pool": goal["best_temporal_rank_full_pool"],
            "best_temporal_score": goal["best_temporal_score"],
            "selection_stage_exclusion": trace["selection_stage_exclusion"],
            "gap_suppressor_group_uid": suppressor,
            "gap_suppressor_rank": suppressor_trace.get("temporal_rank_full_pool", ""),
        })
    miss_diagnostics.sort(key=lambda row: (row["match_id"], int(row["best_temporal_rank_full_pool"])))
    miss_path = EXPERIMENT / "timestamp_rank_miss_diagnostic.csv"
    with miss_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(miss_diagnostics[0]))
        writer.writeheader()
        writer.writerows(miss_diagnostics)
    exclusion_counts = Counter(row["selection_stage_exclusion"] for row in miss_diagnostics)
    high_rank_gap_misses = sum(int(row["best_temporal_rank_full_pool"]) <= 60 and "GAP_45S" in row["selection_stage_exclusion"] for row in miss_diagnostics)

    matched_cases = sum(r["catalog_match_status"] == "MATCHES_CATALOG_GOAL" for r in reconciled)
    recoverable_below_top60 = sum(
        r["catalog_match_status"] == "MATCHES_CATALOG_GOAL" and "RECOVERABLE_GOAL_BELOW_TOP60" in r["review_role"]
        for r in reconciled)
    summary = {
        "scope": "TRAIN only; timestamp reconciliation; no visual overlay inference",
        "cases": len(cases), "cases_matching_catalog_goal_timestamp": matched_cases,
        "cases_without_catalog_goal_timestamp_in_window": len(cases) - matched_cases,
        "distinct_catalog_goals_covered_in_queue": len(matched_catalog_ids),
        "cataloged_goals_in_source": sum(len(items) for items in by_match.values()),
        "recoverable_below_top60_cases_matching_catalog": recoverable_below_top60,
        "recoverable_goals_missed_by_gap45": exclusion_counts.get("GAP_45S_SUPPRESSED_BY_EARLIER_SELECTED_CANDIDATE", 0),
        "recoverable_goals_missed_by_top60_capacity": exclusion_counts.get("TOP60_CAPACITY_OR_AFTER_SHORTLIST_LIMIT", 0),
        "misses_ranked_in_full_pool_top60_but_suppressed_by_gap45": high_rank_gap_misses,
        "miss_exclusion_counts": dict(exclusion_counts),
        "role_case_counts": dict(role_counts), "role_timestamp_match_counts": dict(role_hit_counts),
        "interpretation": "NO_CATALOG_GOAL_IN_WINDOW is not a verified NO_GOAL label; it means only that no annotated catalog timestamp falls inside this window.",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (CASES, GOALS)},
    }
    (EXPERIMENT / "timestamp_reconciliation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Conciliação dos 175 casos com timestamps catalogados",
        "",
        "A decisão GOAL/NO_GOAL não exige inspeção visual quando há timestamp catalogado: a janela do caso pode ser comparada diretamente ao timestamp da partida.",
        "",
        f"- Casos com ao menos um timestamp de gol catalogado dentro da janela 20+40 s: **{matched_cases}/{len(cases)}**.",
        f"- Gols catalogados distintos cobertos por estes casos: **{len(matched_catalog_ids)}**.",
        f"- Casos sem timestamp de gol catalogado dentro da janela: **{len(cases)-matched_cases}**.",
        f"- Gols recuperáveis abaixo do Top-60 identificados por timestamp: **{recoverable_below_top60}**.",
        f"- Dos 21 misses recuperáveis: **{exclusion_counts.get('GAP_45S_SUPPRESSED_BY_EARLIER_SELECTED_CANDIDATE', 0)}** foram suprimidos pelo gap de 45 s e **{exclusion_counts.get('TOP60_CAPACITY_OR_AFTER_SHORTLIST_LIMIT', 0)}** ficaram fora pelo limite de capacidade Top-60.",
        f"- Entre eles, **{high_rank_gap_misses}** tinham rank RF full-pool ≤60 e mesmo assim foram suprimidos pelo gap.",
        "",
        "A ausência de timestamp na janela não confirma NO_GOAL, porque o catálogo pode não anotar todos os eventos. Esses casos permanecem sem rótulo negativo confirmado.",
        "",
        "A análise visual causal BALL/GOAL foi suspensa: os overlays persistidos são amostrados/filtrados, há casos sem CSV e a revisão humana reportou caixas incorretas. Eles não sustentam conclusão de qualidade de detector ou tracking. A conciliação por timestamp e a curva de ranking permanecem utilizáveis.",
        "",
        "Fonte de cobertura: `timestamp_reconciliation.csv`; caminho dos 21 misses: `timestamp_rank_miss_diagnostic.csv`; hashes e contagens: `timestamp_reconciliation.json`.",
    ]
    (EXPERIMENT / "timestamp_reconciliation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
