"""Testa um ranker pairwise L2 em TRAIN, com os 20 controles NO_GOAL revisados."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
DATA = ROOT / "experiments/ball/GOAL-EVENT-RICH-TEMPORAL-DATASET-V1/temporal_group_dataset_v1.csv"
CAPS = (20, 40, 60)
METHODS = ("RF_BASELINE_FROZEN", "PAIR_LTR_BASE36_LOMO", "PAIR_LTR_RICH49_LOMO")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge(rows: list[dict[str, str]]) -> tuple[int, float]:
    merged: list[list[float]] = []
    for start, end in sorted((float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in rows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return len(merged), sum(end - start for start, end in merged)


def select(rows: list[dict[str, str]], score: dict[str, float], k: int) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda r: (-score[r["group_uid"]], r["group_uid"]))
    chosen: list[dict[str, str]] = []
    for row in ranked:
        if all(abs(float(row["best_timestamp_seconds"]) - float(old["best_timestamp_seconds"])) >= 45 for old in chosen):
            chosen.append(row)
        if len(chosen) >= k:
            break
    return chosen


def main() -> int:
    groups = read_csv(DATA)
    trace = read_csv(EXP / "selection_trace.csv")
    controls = read_csv(EXP / "pairwise_control_audit_v1_annotated.csv")
    goals = read_csv(EXP / "case_manifest.csv")
    if len(controls) != 20 or any(r.get("manual_status") != "COMPLETED" or r.get("manual_label") != "NO_GOAL" for r in controls):
        raise ValueError("São necessários os 20 controles humanos concluídos como NO_GOAL.")
    group_by = {r["group_uid"]: r for r in groups}
    trace_by = {r["group_uid"]: r for r in trace}
    categorical = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    rich = [c for c in groups[0] if c not in categorical]
    base = [c for c in rich if not c.startswith("trajectory_")]
    if len(groups) != 5742 or len(base) != 36 or len(rich) != 49 or set(group_by) != set(trace_by):
        raise ValueError("Pool/feature schema divergiu do conjunto TRAIN congelado.")
    matches = sorted({r["match_id"] for r in groups})
    pairs = controls
    predictions = {name: {} for name in METHODS}
    pair_scores = []

    for heldout_match in matches:
        training_pairs = [p for p in pairs if p["match_id"] != heldout_match]
        heldout = [r for r in groups if r["match_id"] == heldout_match]
        predictions["RF_BASELINE_FROZEN"].update({r["group_uid"]: float(trace_by[r["group_uid"]]["temporal_score_lomo"]) for r in heldout})
        for method, feature_names in (("PAIR_LTR_BASE36_LOMO", base), ("PAIR_LTR_RICH49_LOMO", rich)):
            if not training_pairs:
                raise ValueError(f"Sem treino pairwise ao deixar {heldout_match} de fora.")
            endpoint_rows = []
            for pair in training_pairs:
                endpoint_rows.extend((group_by[pair["missed_goal_group_uid"]], group_by[pair["control_group_uid"]]))
            x_endpoints = np.asarray([[float(r[f]) for f in feature_names] for r in endpoint_rows], dtype=float)
            scaler = StandardScaler().fit(x_endpoints)
            differences = []
            for pair in training_pairs:
                pos = scaler.transform(np.asarray([[float(group_by[pair["missed_goal_group_uid"]][f]) for f in feature_names]]))[0]
                neg = scaler.transform(np.asarray([[float(group_by[pair["control_group_uid"]][f]) for f in feature_names]]))[0]
                diff = pos - neg
                differences.extend((diff, -diff))
            x_pair = np.asarray(differences)
            y_pair = np.tile(np.asarray([1, 0], dtype=int), len(training_pairs))
            ranker = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(x_pair, y_pair)
            x_holdout = scaler.transform(np.asarray([[float(r[f]) for f in feature_names] for r in heldout], dtype=float))
            raw = ranker.decision_function(x_holdout)
            predictions[method].update({r["group_uid"]: float(s) for r, s in zip(heldout, raw)})
            pair_scores.append({"heldout_match": heldout_match, "method": method,
                                "training_pairs": len(training_pairs), "features": len(feature_names)})

    pair_rows = []
    score_rows = []
    for method, score_by_uid in predictions.items():
        for uid, value in score_by_uid.items():
            score_rows.append({"method": method, "group_uid": uid,
                               "match_id": group_by[uid]["match_id"], "score": value})
    write_csv(EXP / "pairwise_ltr_hardneg_lomo_scores.csv", score_rows)
    for pair in controls:
        for method in METHODS:
            goal_score = predictions[method][pair["missed_goal_group_uid"]]
            control_score = predictions[method][pair["control_group_uid"]]
            pair_rows.append({"method": method, "goal_catalog_id": pair["goal_catalog_id"], "match_id": pair["match_id"],
                              "missed_goal_group_uid": pair["missed_goal_group_uid"], "control_group_uid": pair["control_group_uid"],
                              "goal_score": goal_score, "reviewed_no_goal_score": control_score,
                              "goal_above_control": goal_score > control_score})
    write_csv(EXP / "pairwise_ltr_hardneg_pair_metrics.csv", pair_rows)

    goal_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in goals:
        goal_match[goal["match_id"]].append(goal)
    summary, by_match = [], []
    for method in METHODS:
        score = predictions[method]
        for k in CAPS:
            covered_ids: set[str] = set()
            cand_total = segment_total = 0
            minutes = 0.0
            no_goal_selected = 0
            for match in matches:
                candidates = [r for r in trace if r["match_id"] == match]
                chosen = select(candidates, score, k)
                segs, secs = merge(chosen)
                local = goal_match[match]
                covered = [g for g in local if any(float(c["delivery_start_seconds"]) <= float(g["goal_timestamp_seconds"]) <= float(c["delivery_end_seconds"]) for c in chosen)]
                covered_ids.update(g["goal_catalog_id"] for g in covered)
                cand_total += len(chosen)
                segment_total += segs
                minutes += secs / 60
                no_goal_selected += sum(any(c["group_uid"] == control_uid for c in chosen)
                                        for control_uid in {p["control_group_uid"] for p in controls if p["match_id"] == match})
                by_match.append({"method": method, "k": k, "match_id": match, "goals": len(local), "covered_goals": len(covered),
                                 "candidates": len(chosen), "segments": segs, "unique_minutes": round(secs / 60, 3)})
            summary.append({"method": method, "k": k, "covered_goals": len(covered_ids), "catalog_goals": len(goals),
                            "recall": round(len(covered_ids) / len(goals), 6), "candidates": cand_total,
                            "segments": segment_total, "unique_minutes_total": round(minutes, 3),
                            "unique_minutes_per_match": round(minutes / len(matches), 3),
                            "reviewed_no_goal_controls_selected": no_goal_selected})
    write_csv(EXP / "pairwise_ltr_hardneg_summary.csv", summary)
    write_csv(EXP / "pairwise_ltr_hardneg_by_match.csv", by_match)

    pair_summary = []
    for method in METHODS:
        local = [r for r in pair_rows if r["method"] == method]
        wins = sum(bool(r["goal_above_control"]) for r in local)
        pair_summary.append({"method": method, "pairs": len(local), "goal_above_control": wins,
                             "pairwise_accuracy": round(wins / len(local), 4)})
    md = [
        "# Ranker pairwise com hard negatives confirmados — TRAIN LOMO",
        "",
        "Foram usados somente os pares confirmados pelo revisor: 20 comparadores NO_GOAL e seus grupos positivos associados aos timestamps. Os demais weak_label=0 não entraram no treino. Em cada dobra, pares de uma partida inteira foram excluídos; cada modelo restante pontuou todo o pool daquela partida.",
        "",
        "Logística pairwise L2 (C=1) treinada sobre diferenças de features, comparada com score RF congelado. Sem busca de hiperparâmetros. O dataset rico contém as 36 features atuais e 13 resumos de tendência temporal.",
        "",
        "## Ordenação dos pares revisados",
        "",
        "| Método | Pares | Gols acima do controle | Acurácia pareada |",
        "|---|---:|---:|---:|",
    ]
    for r in pair_summary:
        md.append(f"| {r['method']} | {r['pairs']} | {r['goal_above_control']} | {r['pairwise_accuracy']:.1%} |")
    md.extend(["", "## Cobertura e custo sob o mesmo gap45", "",
               "| Método | K | Gols cobertos | Recall | Min únicos/jogo | Segmentos | Controles NO_GOAL selecionados |",
               "|---|---:|---:|---:|---:|---:|---:|"])
    for r in summary:
        md.append(f"| {r['method']} | {r['k']} | {r['covered_goals']}/{r['catalog_goals']} | {float(r['recall']):.1%} | {r['unique_minutes_per_match']:.3f} | {r['segments']} | {r['reviewed_no_goal_controls_selected']} |")
    md.extend([
        "",
        "## Limitações",
        "",
        "A validação é exploratória e limitada a 20 hard negatives selecionados entre os candidatos de maior rank; há apenas 11 partidas com controle revisado. A avaliação pareada é mais informativa que acurácia binária global, mas não representa o universo de falsos candidatos. A redução de controles no Top-K deve ser lida junto com a perda/ganho de Recall@K; não escolher o método com base só nos 20 pares. Não usar VALIDATION/TEST para selecionar hiperparâmetros.",
        "",
        "`inside_goal_fraction` continua sendo fração de detecções dentro da bbox expandida; não é duração física dentro da abertura do gol. Os 13 sinais extras descrevem tendência das mesmas deteções e não corrigem identidade de bola ou caixa de goleira.",
    ])
    (EXP / "pairwise_ltr_hardneg.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(f"{r['method']}: {r['goal_above_control']}/{r['pairs']} pairs" for r in pair_summary))
    print("K60:")
    for r in summary:
        if r["k"] == 60:
            print(f"{r['method']}: {r['covered_goals']}/{r['catalog_goals']}, {r['unique_minutes_per_match']} min/match, {r['reviewed_no_goal_controls_selected']} reviewed NO_GOAL selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
