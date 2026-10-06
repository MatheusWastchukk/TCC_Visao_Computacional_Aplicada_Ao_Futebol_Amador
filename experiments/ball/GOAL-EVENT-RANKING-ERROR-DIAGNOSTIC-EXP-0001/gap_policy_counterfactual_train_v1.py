"""Simula políticas gap na seleção RF LOMO congelada, somente com timestamps TRAIN."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
TRACE = EXP / "selection_trace.csv"
GOAL_MANIFEST = EXP / "case_manifest.csv"
BASELINE = EXP / "coverage_baseline.csv"
GAPS = (0, 15, 30, 45, 60, 90)
CAP_PER_MATCH = 60


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def merge_intervals(rows: list[dict[str, str]]) -> tuple[int, float]:
    intervals = sorted((float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in rows)
    if not intervals:
        return 0, 0.0
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return len(merged), sum(end - start for start, end in merged)


def main() -> int:
    trace = read_csv(TRACE)
    catalog_rows = read_csv(GOAL_MANIFEST)
    old_baseline = next(row for row in read_csv(BASELINE)
                        if row["method"] == "TEMPORAL_RF_LOMO_GAP45" and row["k"] == "60")
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        by_match[row["match_id"]].append(row)
    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in catalog_rows:
        goals_by_match[goal["match_id"]].append(goal)
    baseline_miss_ids = {g["goal_catalog_id"] for g in catalog_rows
                         if g["pool_recoverable"] == "True" and g["covered_by_existing_top60"] == "False"}

    rows_out: list[dict[str, object]] = []
    match_out: list[dict[str, object]] = []
    goal_out: list[dict[str, object]] = []
    for gap in GAPS:
        selected_by_match: dict[str, list[dict[str, str]]] = {}
        covered_goal_ids: set[str] = set()
        for match, candidates in by_match.items():
            ranked = sorted(candidates, key=lambda r: (-float(r["temporal_score_lomo"]), r["group_uid"]))
            selected: list[dict[str, str]] = []
            for candidate in ranked:
                center = float(candidate["best_timestamp_seconds"])
                if all(abs(center - float(existing["best_timestamp_seconds"])) >= gap for existing in selected):
                    selected.append(candidate)
                if len(selected) == CAP_PER_MATCH:
                    break
            selected_by_match[match] = selected
            intervals = [(float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in selected]
            local_covered = [g for g in goals_by_match[match]
                             if any(start <= float(g["goal_timestamp_seconds"]) <= end for start, end in intervals)]
            local_recoverable = [g for g in goals_by_match[match] if g["pool_recoverable"] == "True"]
            local_recoverable_covered = [g for g in local_recoverable if g in local_covered]
            covered_goal_ids.update(g["goal_catalog_id"] for g in local_covered)
            for goal in goals_by_match[match]:
                goal_timestamp = float(goal["goal_timestamp_seconds"])
                covering = [(rank, candidate) for rank, candidate in enumerate(selected, 1)
                            if float(candidate["delivery_start_seconds"]) <= goal_timestamp <= float(candidate["delivery_end_seconds"])]
                best_rank, best_candidate = min(covering, key=lambda pair: pair[0]) if covering else ("", {})
                goal_out.append({
                    "gap_seconds": gap, "goal_catalog_id": goal["goal_catalog_id"], "match_id": match,
                    "goal_timestamp_seconds": goal["goal_timestamp_seconds"],
                    "pool_recoverable": goal["pool_recoverable"],
                    "covered_by_existing_gap45_top60": goal["covered_by_existing_top60"],
                    "covered_by_counterfactual": bool(covering),
                    "best_selected_rank_for_goal": best_rank,
                    "best_selected_group_uid": best_candidate.get("group_uid", ""),
                })
            segments, minutes = merge_intervals(selected)
            match_out.append({
                "gap_seconds": gap, "match_id": match,
                "selected_candidates": len(selected), "merged_delivery_segments": segments,
                "unique_delivery_minutes": round(minutes / 60, 3),
                "catalog_goals": len(goals_by_match[match]), "covered_catalog_goals": len(local_covered),
                "pool_recoverable_goals": len(local_recoverable),
                "covered_pool_recoverable_goals": len(local_recoverable_covered),
            })
        total_candidates = sum(len(items) for items in selected_by_match.values())
        total_segments = 0
        total_unique_minutes = 0.0
        for selected in selected_by_match.values():
            segments, seconds = merge_intervals(selected)
            total_segments += segments
            total_unique_minutes += seconds / 60
        goal_total = sum(len(items) for items in goals_by_match.values())
        recoverable_goal_ids = {g["goal_catalog_id"] for g in catalog_rows if g["pool_recoverable"] == "True"}
        rows_out.append({
            "experiment": "GAP_POLICY_COUNTERFACTUAL_TRAIN_V1",
            "score_source": "frozen RF LOMO temporal scores",
            "gap_seconds": gap, "candidate_cap_per_match": CAP_PER_MATCH,
            "selected_candidates_total": total_candidates, "merged_delivery_segments_total": total_segments,
            "unique_delivery_minutes_total": round(total_unique_minutes, 3),
            "unique_delivery_minutes_per_match": round(total_unique_minutes / len(by_match), 3),
            "catalog_goals": goal_total, "covered_catalog_goals": len(covered_goal_ids),
            "baseline_recoverable_misses_recovered": len(covered_goal_ids & baseline_miss_ids),
            "recall_catalog": round(len(covered_goal_ids) / goal_total, 6),
            "pool_recoverable_goals": len(recoverable_goal_ids),
            "covered_pool_recoverable_goals": len(covered_goal_ids & recoverable_goal_ids),
            "recall_conditional_recoverable": round(len(covered_goal_ids & recoverable_goal_ids) / len(recoverable_goal_ids), 6),
            "test_or_validation_used": False,
        })

    baseline_45 = next(row for row in rows_out if row["gap_seconds"] == 45)
    # Reproduction guard: the gap45 implementation must match the persisted operational baseline.
    actual = (baseline_45["selected_candidates_total"], baseline_45["merged_delivery_segments_total"],
              round(float(baseline_45["unique_delivery_minutes_total"]), 3), baseline_45["covered_catalog_goals"])
    expected = (int(old_baseline["selected_candidates"]), int(old_baseline["merged_delivery_segments"]),
                round(float(old_baseline["unique_review_minutes_total"]), 3), int(old_baseline["covered_goals"]))
    if actual != expected:
        raise ValueError(f"gap45 divergiu do baseline preservado: atual={actual}, esperado={expected}")

    for name, data in (("gap_policy_counterfactual_summary.csv", rows_out),
                       ("gap_policy_counterfactual_by_match.csv", match_out),
                       ("gap_policy_counterfactual_by_goal.csv", goal_out)):
        path = EXP / name
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    paired_rows = []
    match_lookup = {(int(row["gap_seconds"]), row["match_id"]): row for row in match_out}
    for from_gap, to_gap in ((30, 45), (45, 60)):
        for match in sorted(by_match):
            before, after = match_lookup[(from_gap, match)], match_lookup[(to_gap, match)]
            paired_rows.append({
                "comparison": f"GAP{from_gap}_TO_GAP{to_gap}", "match_id": match,
                "delta_covered_catalog_goals": int(after["covered_catalog_goals"]) - int(before["covered_catalog_goals"]),
                "delta_covered_recoverable_goals": int(after["covered_pool_recoverable_goals"]) - int(before["covered_pool_recoverable_goals"]),
                "delta_candidates": int(after["selected_candidates"]) - int(before["selected_candidates"]),
                "delta_merged_segments": int(after["merged_delivery_segments"]) - int(before["merged_delivery_segments"]),
                "delta_unique_minutes": round(float(after["unique_delivery_minutes"]) - float(before["unique_delivery_minutes"]), 3),
            })
    paired_path = EXP / "gap_policy_paired_match_diagnostic.csv"
    with paired_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    summary = {
        "scope": "TRAIN-only deterministic selection simulation; TEST and VALIDATION not read",
        "gaps_seconds": list(GAPS), "candidate_cap_per_match": CAP_PER_MATCH,
        "baseline_gap45_reproduced": True, "operational_baseline": actual,
        "interpretation_limit": "Exploratory within the existing TRAIN set; not a selected or promoted operating policy.",
    }
    (EXP / "gap_policy_counterfactual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Contrafactual da política gap — TRAIN",
        "",
        "Scores RF LOMO, candidatos, janela 20+40 s e teto de 60 candidatos por partida foram mantidos. Só varia o afastamento temporal mínimo do seletor. A regra aceita um grupo se sua distância para todos os selecionados anteriores for pelo menos o gap.",
        "",
        "O ponto gap45 precisa reproduzir o baseline persistido exatamente; caso contrário, o script interrompe antes de aceitar o sweep.",
        "",
        "| Gap | Candidatos | Segmentos unidos | Minutos únicos/jogo | Gols cobertos | Misses baseline recuperados/21 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_out:
        md.append(f"| {row['gap_seconds']} s | {row['selected_candidates_total']} | {row['merged_delivery_segments_total']} | {row['unique_delivery_minutes_per_match']:.3f} | {row['covered_catalog_goals']}/152 ({row['recall_catalog']:.1%}) | {row['baseline_recoverable_misses_recovered']}/21 |")
    pair_counts = {}
    for comparison in ("GAP30_TO_GAP45", "GAP45_TO_GAP60"):
        paired = [row for row in paired_rows if row["comparison"] == comparison]
        pair_counts[comparison] = {
            "wins": sum(int(row["delta_covered_catalog_goals"]) > 0 for row in paired),
            "losses": sum(int(row["delta_covered_catalog_goals"]) < 0 for row in paired),
            "ties": sum(int(row["delta_covered_catalog_goals"]) == 0 for row in paired),
            "catalog_goal_delta": sum(int(row["delta_covered_catalog_goals"]) for row in paired),
            "candidate_delta": sum(int(row["delta_candidates"]) for row in paired),
            "segment_delta": sum(int(row["delta_merged_segments"]) for row in paired),
            "unique_minutes_delta": round(sum(float(row["delta_unique_minutes"]) for row in paired), 3),
        }
    md.extend([
        "",
        f"Pareado por partida: gap30→45 teve +1 gol em {pair_counts['GAP30_TO_GAP45']['wins']} partidas, −1 em {pair_counts['GAP30_TO_GAP45']['losses']} e empate em {pair_counts['GAP30_TO_GAP45']['ties']}; delta agregado {pair_counts['GAP30_TO_GAP45']['catalog_goal_delta']} gols, {pair_counts['GAP30_TO_GAP45']['candidate_delta']} candidatos, {pair_counts['GAP30_TO_GAP45']['segment_delta']} segmentos e +{pair_counts['GAP30_TO_GAP45']['unique_minutes_delta']:.3f} minutos únicos.",
        f"Gap45→60 teve +cobertura em {pair_counts['GAP45_TO_GAP60']['wins']} partidas, −cobertura em {pair_counts['GAP45_TO_GAP60']['losses']} e empate em {pair_counts['GAP45_TO_GAP60']['ties']}; delta agregado {pair_counts['GAP45_TO_GAP60']['catalog_goal_delta']} gols, {pair_counts['GAP45_TO_GAP60']['candidate_delta']} candidatos, {pair_counts['GAP45_TO_GAP60']['segment_delta']} segmentos e {pair_counts['GAP45_TO_GAP60']['unique_minutes_delta']:.3f} minutos únicos.",
        "",
        "Decisão: manter gap45 como referência congelada. A alternativa gap30 reduz minutos únicos, mas a diferença de cobertura não é consistente por partida e também muda o volume de candidatos/segmentos; gap60 tampouco oferece ganho operacional claro. Nenhuma política é promovida com este sweep TRAIN-only.",
        "",
        "Compare cobertura dos timestamps catalogados com candidatos, segmentos e minutos únicos. Minutos únicos são duração de vídeo entregue, não tempo humano. Estes resultados são exploratórios TRAIN e não selecionam uma política final.",
        "",
        "Resultados: `gap_policy_counterfactual_summary.csv`, detalhe por partida em `gap_policy_counterfactual_by_match.csv`, pareamento em `gap_policy_paired_match_diagnostic.csv` e por gol em `gap_policy_counterfactual_by_goal.csv`; `VALIDATION` e `TEST` não são consultados.",
    ])
    (EXP / "gap_policy_counterfactual_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_gap45": baseline_45, "gap_sweep": rows_out}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
