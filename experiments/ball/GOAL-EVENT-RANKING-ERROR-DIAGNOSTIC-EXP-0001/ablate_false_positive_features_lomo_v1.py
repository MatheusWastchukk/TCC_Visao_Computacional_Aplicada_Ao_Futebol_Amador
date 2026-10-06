"""Ablação exploratória TRAIN LOMO das features associadas aos 20 falsos controles."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
DATA = ROOT / "experiments/ball/GOAL-EVENT-RICH-TEMPORAL-DATASET-V1/temporal_group_dataset_v1.csv"
GAP = 45
K = 60


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select(rows: list[dict[str, str]], scores: dict[str, float]) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda r: (-scores[r["group_uid"]], r["group_uid"])):
        t = float(row["best_timestamp_seconds"])
        if all(abs(t - float(old["best_timestamp_seconds"])) >= GAP for old in chosen):
            chosen.append(row)
        if len(chosen) == K:
            break
    return chosen


def main() -> int:
    groups = read_csv(DATA)
    goals = read_csv(EXP / "case_manifest.csv")
    trace = read_csv(EXP / "selection_trace.csv")
    controls = read_csv(EXP / "pairwise_control_audit_v1_annotated.csv")
    if len(controls) != 20 or any(r.get("manual_status") != "COMPLETED" or r.get("manual_label") != "NO_GOAL" for r in controls):
        raise ValueError("Exige os 20 controles humanos confirmados NO_GOAL.")
    by_uid = {r["group_uid"]: r for r in groups}
    if len(by_uid) != len(groups) or len(groups) != 5742:
        raise ValueError("O pool não coincide com o dataset TRAIN congelado esperado.")
    categorical = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    features = [c for c in groups[0] if c not in categorical]
    base_features = [c for c in features if not c.startswith("trajectory_")]
    variants = {
        "LR_BASE36": list(base_features),
        "LR_SEM_INSIDE": [f for f in base_features if f != "inside_goal_fraction" and not (f.startswith("bin_") and f.endswith("_inside_fraction"))],
        "LR_SEM_CONTAGEM": [f for f in base_features if f not in {"hit_count", "unique_timestamps", "duration_seconds", "gap_count"}],
        "LR_SEM_INSIDE_NEM_CONTAGEM": [f for f in base_features if f != "inside_goal_fraction" and not (f.startswith("bin_") and f.endswith("_inside_fraction")) and f not in {"hit_count", "unique_timestamps", "duration_seconds", "gap_count"}],
    }
    positives = {r["best_temporal_group_uid"] for r in goals if r["pool_recoverable"].casefold() == "true" and r["best_temporal_group_uid"]}
    negatives = {r["control_group_uid"] for r in controls}
    if positives & negatives:
        raise ValueError("Grupo tem rótulos positivos e negativos.")
    match_labels: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for uid in positives:
        match_labels[by_uid[uid]["match_id"]].append((uid, 1))
    for uid in negatives:
        match_labels[by_uid[uid]["match_id"]].append((uid, 0))
    matches = sorted({r["match_id"] for r in groups})
    all_scores: dict[str, dict[str, float]] = {name: {} for name in variants}
    for heldout in matches:
        train = [(uid, label) for match, rows in match_labels.items() if match != heldout for uid, label in rows]
        test_rows = [r for r in groups if r["match_id"] == heldout]
        y = np.asarray([label for _, label in train], dtype=int)
        if set(y.tolist()) != {0, 1}:
            raise ValueError(f"Dobra sem duas classes: {heldout}")
        for name, cols in variants.items():
            x_train = np.asarray([[float(by_uid[uid][c]) for c in cols] for uid, _ in train], dtype=float)
            x_test = np.asarray([[float(r[c]) for c in cols] for r in test_rows], dtype=float)
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42))
            model.fit(x_train, y)
            scores = model.decision_function(x_test)
            all_scores[name].update({r["group_uid"]: float(s) for r, s in zip(test_rows, scores)})

    summaries = []
    score_rows = []
    pair_rows = []
    for name, scores in all_scores.items():
        for uid, score in scores.items():
            score_rows.append({"method": name, "group_uid": uid, "match_id": by_uid[uid]["match_id"], "score": score})
        covered: set[str] = set()
        selected_controls = 0
        for match in matches:
            selected = select([r for r in trace if r["match_id"] == match], scores)
            intervals = [(float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in selected]
            covered.update(g["goal_catalog_id"] for g in goals if g["match_id"] == match and
                           any(a <= float(g["goal_timestamp_seconds"]) <= b for a, b in intervals))
            selected_controls += sum(any(r["group_uid"] == uid for r in selected) for uid in negatives if by_uid[uid]["match_id"] == match)
        wins = 0
        for control in controls:
            pos = scores[control["missed_goal_group_uid"]]
            neg = scores[control["control_group_uid"]]
            correct = pos > neg
            wins += correct
            pair_rows.append({"method": name, "goal_catalog_id": control["goal_catalog_id"], "match_id": control["match_id"], "goal_above_control": correct})
        summaries.append({"method": name, "features": len(variants[name]), "goal_above_control": wins, "pairs": len(controls),
                          "pairwise_accuracy": round(wins / len(controls), 4), "catalog_goals_covered_k60": len(covered),
                          "catalog_goals": len(goals), "recall_k60": round(len(covered) / len(goals), 4),
                          "reviewed_no_goal_controls_selected_k60": selected_controls})
    write_csv(EXP / "false_positive_feature_ablation_summary.csv", summaries)
    write_csv(EXP / "false_positive_feature_ablation_scores.csv", score_rows)
    write_csv(EXP / "false_positive_feature_ablation_pairs.csv", pair_rows)
    md = ["# Ablação de features associadas a falsos positivos — TRAIN LOMO", "",
          "Classificadores LR L2 (C=0,1), mesmo protocolo por partida, mesmos 149 positivos e 20 hard negatives confirmados. Somente as colunas foram removidas; nenhum hiperparâmetro foi ajustado. Recall@60 é cobertura dos 152 timestamps com seleção por gap45. Diagnóstico exploratório: os mesmos 20 pares também foram usados para treino em dobras diferentes, portanto não escolher variante por pairwise accuracy.", "",
          "| Variante | Features | Pares corretos | Recall@60 | NO_GOAL selecionados |", "|---|---:|---:|---:|---:|"]
    for r in summaries:
        md.append(f"| {r['method']} | {r['features']} | {r['goal_above_control']}/{r['pairs']} ({r['pairwise_accuracy']:.1%}) | {r['catalog_goals_covered_k60']}/{r['catalog_goals']} ({r['recall_k60']:.1%}) | {r['reviewed_no_goal_controls_selected_k60']} |")
    md.extend(["", "As ablações indicam dependência de algumas famílias de features, mas não isolam causalidade nem constituem método novo. Nenhuma variante é promovida; VALIDATION e TEST não foram usados."])
    (EXP / "false_positive_feature_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for row in summaries:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
