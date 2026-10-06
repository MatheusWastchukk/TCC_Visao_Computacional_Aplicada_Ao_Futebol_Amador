"""Mede a curva de sensibilidade ao misturar RF congelado e sinais pairwise LOMO."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
KS = (20, 40, 60)
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def intervals(selected: list[dict[str, str]]) -> tuple[int, float]:
    merged: list[list[float]] = []
    for start, end in sorted((float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in selected):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return len(merged), sum(end - start for start, end in merged)


def select(rows: list[dict[str, str]], score: dict[str, float], k: int) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda r: (-score[r["group_uid"]], r["group_uid"]))
    selected: list[dict[str, str]] = []
    for row in ranked:
        if all(abs(float(row["best_timestamp_seconds"]) - float(previous["best_timestamp_seconds"])) >= 45 for previous in selected):
            selected.append(row)
        if len(selected) >= k:
            break
    return selected


def main() -> int:
    trace = read_csv(EXP / "selection_trace.csv")
    goals = read_csv(EXP / "case_manifest.csv")
    controls = read_csv(EXP / "pairwise_control_audit_v1_annotated.csv")
    model_scores = read_csv(EXP / "pairwise_ltr_hardneg_lomo_scores.csv")
    if len(controls) != 20 or any(r.get("manual_status") != "COMPLETED" or r.get("manual_label") != "NO_GOAL" for r in controls):
        raise ValueError("A análise exige os 20 controles humanos NO_GOAL.")
    groups_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        groups_by_match[row["match_id"]].append(row)
    for row in goals:
        goals_by_match[row["match_id"]].append(row)
    learned: dict[tuple[str, str], float] = {(row["method"], row["group_uid"]): float(row["score"]) for row in model_scores}
    baseline: dict[str, float] = {row["group_uid"]: float(row["temporal_score_lomo"]) for row in trace}

    # Converter scores heterogêneos em utilidade de rank dentro de cada partida.
    utilities: dict[tuple[str, str], float] = {}
    for match, rows in groups_by_match.items():
        n = len(rows)
        for name, raw in (("RF_BASELINE_FROZEN", baseline),
                          ("PAIR_LTR_BASE36_LOMO", {r["group_uid"]: learned[("PAIR_LTR_BASE36_LOMO", r["group_uid"])] for r in rows}),
                          ("PAIR_LTR_RICH49_LOMO", {r["group_uid"]: learned[("PAIR_LTR_RICH49_LOMO", r["group_uid"])] for r in rows})):
            ranked = sorted(rows, key=lambda row: (-raw[row["group_uid"]], row["group_uid"]))
            rank_by_uid = {row["group_uid"]: rank for rank, row in enumerate(ranked)}
            for row in rows:
                rank = rank_by_uid[row["group_uid"]]
                utilities[(name, row["group_uid"])] = 1.0 - rank / max(1, n - 1)

    evaluated: dict[str, dict[str, float]] = {"RF_BASELINE_FROZEN": baseline}
    for source in ("PAIR_LTR_BASE36_LOMO", "PAIR_LTR_RICH49_LOMO"):
        source_name = "PAIR_LTR_BASE36_LOMO" if source.startswith("PAIR_LTR_BASE36") else "PAIR_LTR_RICH49_LOMO"
        for alpha in ALPHAS:
            name = f"BLEND_{source_name}_A{alpha:.2f}"
            evaluated[name] = {
                row["group_uid"]: (1 - alpha) * utilities[("RF_BASELINE_FROZEN", row["group_uid"])]
                + alpha * utilities[(source_name, row["group_uid"])]
                for row in trace
            }

    rows_out = []
    pair_out = []
    for name, score in evaluated.items():
        wins = 0
        for control in controls:
            goal_score = score[control["missed_goal_group_uid"]]
            no_goal_score = score[control["control_group_uid"]]
            above = goal_score > no_goal_score
            wins += above
            pair_out.append({"method": name, "goal_catalog_id": control["goal_catalog_id"], "match_id": control["match_id"],
                             "goal_above_no_goal": above, "goal_score": goal_score, "no_goal_score": no_goal_score})
        for k in KS:
            covered_ids: set[str] = set()
            minutes = 0.0
            segment_count = selected_count = no_goal_selected = 0
            for match, candidates in groups_by_match.items():
                selected = select(candidates, score, k)
                selected_count += len(selected)
                segs, seconds = intervals(selected)
                segment_count += segs
                minutes += seconds / 60
                no_goal_selected += sum(any(r["group_uid"] == c["control_group_uid"] for r in selected)
                                        for c in controls if c["match_id"] == match)
                for goal in goals_by_match[match]:
                    if any(float(row["delivery_start_seconds"]) <= float(goal["goal_timestamp_seconds"]) <= float(row["delivery_end_seconds"]) for row in selected):
                        covered_ids.add(goal["goal_catalog_id"])
            rows_out.append({"method": name, "k": k, "pairwise_wins": wins, "pairwise_total": len(controls),
                             "pairwise_accuracy": round(wins / len(controls), 4), "covered_goals": len(covered_ids),
                             "catalog_goals": len(goals), "recall": round(len(covered_ids) / len(goals), 6),
                             "selected_candidates": selected_count, "segments": segment_count,
                             "unique_minutes_total": round(minutes, 3),
                             "unique_minutes_per_match": round(minutes / len(groups_by_match), 3),
                             "reviewed_no_goal_controls_selected": no_goal_selected})
    write_csv(EXP / "pairwise_rank_blend_frontier.csv", rows_out)
    write_csv(EXP / "pairwise_rank_blend_pairwise.csv", pair_out)
    md = [
        "# Fronteira contrafactual de mistura do ranker — TRAIN",
        "",
        "A escala dos scores foi normalizada por percentil de rank dentro de cada partida. Alpha controla a fração do ranker pairwise LOMO somada ao RF congelado: alpha=0 reproduz o RF e alpha=1 usa somente o pairwise. Foram avaliados valores fixos, sem escolher um vencedor para integração.",
        "",
        "Esta análise mede sensibilidade nos mesmos 20 hard negatives usados para treinar/avaliar o ranker; a acurácia pareada é otimista para esses casos e não representa novos controles independentes.",
        "",
        "| Método | K | Pares corretos | Recall catálogo | Minutos únicos/jogo | NO_GOAL revisados no Top-K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows_out:
        if row["k"] in {20, 60}:
            md.append(f"| {row['method']} | {row['k']} | {row['pairwise_wins']}/{row['pairwise_total']} | {float(row['recall']):.1%} | {row['unique_minutes_per_match']:.3f} | {row['reviewed_no_goal_controls_selected']} |")
    md.extend(["", "Nenhum blend foi promovido. VALIDATION e TEST continuam fechados."])
    (EXP / "pairwise_rank_blend_frontier.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((EXP / "pairwise_rank_blend_frontier.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
