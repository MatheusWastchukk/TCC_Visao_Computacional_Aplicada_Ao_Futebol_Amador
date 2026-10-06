"""Avalia reranking TRAIN com gols catalogados positivos e controles humanos NO_GOAL."""
from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
FEATURE_DATA = ROOT / "experiments/ball/GOAL-EVENT-RICH-TEMPORAL-DATASET-V1/temporal_group_dataset_v1.csv"
METHODS = ("RF_BASELINE_FROZEN", "LR_BASE36_HARDNEG_LOMO", "LR_RICH49_HARDNEG_LOMO", "RF_RICH49_HARDNEG_LOMO", "XGB_RICH49_HARDNEG_LOMO")
CAPS = (20, 40, 60)
GAP_SECONDS = 45


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_intervals(rows: list[dict[str, str]]) -> tuple[int, float]:
    intervals = sorted((float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in rows)
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return len(merged), sum(end - start for start, end in merged)


def make_model(method: str):
    if method.startswith("LR_"):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42))
    if method.startswith("RF_"):
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1))
    if method.startswith("XGB_"):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
                             eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=1)
    raise ValueError(method)


def score_model(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "steps") and isinstance(model[-1], LogisticRegression):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]

def select(rows: list[dict[str, str]], score_by_group: dict[str, float], cap: int) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda r: (-score_by_group[r["group_uid"]], r["group_uid"]))
    selected: list[dict[str, str]] = []
    for row in ranked:
        center = float(row["best_timestamp_seconds"])
        if all(abs(center - float(previous["best_timestamp_seconds"])) >= GAP_SECONDS for previous in selected):
            selected.append(row)
        if len(selected) >= cap:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=EXP / "pairwise_control_audit_v2_batch1_annotated.csv")
    parser.add_argument("--prior-controls", type=Path, default=EXP / "pairwise_control_audit_v1_annotated.csv")
    parser.add_argument("--output", type=Path, default=EXP / "PROGRESSIVE-HARDNEG-EXP-0001-BATCH1")
    args = parser.parse_args()
    groups = read_csv(FEATURE_DATA)
    goals = read_csv(EXP / "case_manifest.csv")
    trace = read_csv(EXP / "selection_trace.csv")
    prior_controls = read_csv(args.prior_controls) if args.prior_controls else []
    batch_rows = read_csv(args.controls)
    if any(row.get("manual_status") != "COMPLETED" or row.get("manual_label") != "NO_GOAL" for row in prior_controls):
        raise ValueError("Prior controls must all be human-confirmed NO_GOAL rows.")
    allowed_labels = {"", "GOAL", "NO_GOAL", "NOT_EVALUABLE"}
    if any(row.get("manual_label", "") not in allowed_labels for row in batch_rows):
        raise ValueError("Unexpected manual label in progressive batch.")
    batch_no_goals = [row for row in batch_rows
                      if row.get("manual_status") == "COMPLETED" and row.get("manual_label") == "NO_GOAL"]
    controls = [*prior_controls, *batch_no_goals]
    group_by_uid = {row["group_uid"]: row for row in groups}
    trace_by_uid = {row["group_uid"]: row for row in trace}
    if not controls:
        raise ValueError("No human-confirmed NO_GOAL rows are available for training.")
    negative_uids_check = [row["control_group_uid"] for row in controls]
    if len(set(negative_uids_check)) != len(negative_uids_check):
        raise ValueError("Duplicate control group in annotation file.")
    batch_review_counts = {
        "candidates": len(batch_rows),
        "new_no_goal_confirmed": len(batch_no_goals),
        "goal": sum(row.get("manual_status") == "COMPLETED" and row.get("manual_label") == "GOAL" for row in batch_rows),
        "not_evaluable": sum(row.get("manual_status") == "COMPLETED" and row.get("manual_label") == "NOT_EVALUABLE" for row in batch_rows),
        "pending": sum(row.get("manual_status") != "COMPLETED" for row in batch_rows),
    }
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if len(group_by_uid) != len(groups) or set(group_by_uid) != set(trace_by_uid):
        raise ValueError("Dataset de features e selection_trace não têm o mesmo conjunto de grupos.")

    categorical = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    all_features = [key for key in groups[0] if key not in categorical]
    base_features = [key for key in all_features if not key.startswith("trajectory_")]
    rich_features = all_features
    if len(base_features) != 36 or len(rich_features) != 49:
        raise ValueError(f"Contagem de features inesperada: base={len(base_features)} rich={len(rich_features)}")
    matrix = {row["group_uid"]: row for row in groups}
    matches = sorted({row["match_id"] for row in groups})

    # Positivos são somente grupos associados a timestamps catalogados e recuperáveis.
    positive_uids = {row["best_temporal_group_uid"] for row in goals
                     if row["pool_recoverable"].casefold() == "true" and row["best_temporal_group_uid"]}
    negative_uids = {row["control_group_uid"] for row in controls}
    if positive_uids & negative_uids:
        raise ValueError("Há grupo rotulado simultaneamente positivo por timestamp e negativo pela revisão.")
    if not positive_uids <= set(group_by_uid) or not negative_uids <= set(group_by_uid):
        raise ValueError("Faltam grupos rotulados no dataset temporal de features.")

    labels_by_match: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for uid in positive_uids:
        labels_by_match[group_by_uid[uid]["match_id"]].append((uid, 1))
    for uid in negative_uids:
        labels_by_match[group_by_uid[uid]["match_id"]].append((uid, 0))

    predictions: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    for match in matches:
        heldout = [row for row in groups if row["match_id"] == match]
        train_samples = [(uid, label) for train_match, samples in labels_by_match.items()
                         if train_match != match for uid, label in samples]
        if len({label for _, label in train_samples}) < 2:
            raise ValueError(f"Fold sem as duas classes ao deixar {match} de fora.")
        for method in METHODS[1:]:
            feature_names = base_features if method == "LR_BASE36_HARDNEG_LOMO" else rich_features
            x_train = np.asarray([[float(group_by_uid[uid][name]) for name in feature_names]
                                  for uid, _ in train_samples], dtype=float)
            y_train = np.asarray([label for _, label in train_samples], dtype=int)
            x_test = np.asarray([[float(row[name]) for name in feature_names] for row in heldout], dtype=float)
            model = make_model(method)
            if method.startswith("XGB_"):
                class_counts = np.bincount(y_train, minlength=2)
                sample_weights = np.asarray([len(y_train) / (2 * class_counts[label]) for label in y_train])
                model.fit(x_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(x_train, y_train)
            scores = score_model(model, x_test)
            predictions[method].update({row["group_uid"]: float(score) for row, score in zip(heldout, scores)})
        predictions["RF_BASELINE_FROZEN"].update({
            row["group_uid"]: float(trace_by_uid[row["group_uid"]]["temporal_score_lomo"]) for row in heldout
        })

    score_rows: list[dict[str, object]] = []
    for method, scores in predictions.items():
        for uid, value in scores.items():
            score_rows.append({"method": method, "group_uid": uid, "match_id": group_by_uid[uid]["match_id"], "score": value})

    selected_by_method_cap: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    summary_rows: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    for method in METHODS:
        scores = predictions[method]
        for cap in CAPS:
            covered_ids: set[str] = set()
            candidate_total = segment_total = 0
            minutes_total = 0.0
            for match in matches:
                match_rows = [row for row in trace if row["match_id"] == match]
                selected = select(match_rows, scores, cap)
                selected_by_method_cap[(method, cap, match)] = selected
                segments, seconds = merge_intervals(selected)
                candidate_total += len(selected)
                segment_total += segments
                minutes_total += seconds / 60
                intervals = [(float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in selected]
                local_goals = [g for g in goals if g["match_id"] == match]
                covered = [g for g in local_goals if any(a <= float(g["goal_timestamp_seconds"]) <= b for a, b in intervals)]
                covered_ids.update(g["goal_catalog_id"] for g in covered)
                per_match_rows.append({
                    "method": method, "k": cap, "match_id": match,
                    "catalog_goals": len(local_goals), "covered_catalog_goals": len(covered),
                    "selected_candidates": len(selected), "merged_delivery_segments": segments,
                    "unique_delivery_minutes": round(seconds / 60, 3),
                })
            summary_rows.append({
                "method": method, "k": cap, "gap_seconds": GAP_SECONDS,
                "catalog_goals": len(goals), "covered_catalog_goals": len(covered_ids),
                "catalog_recall": round(len(covered_ids) / len(goals), 6),
                "pool_recoverable_goals": sum(g["pool_recoverable"].casefold() == "true" for g in goals),
                "covered_pool_recoverable_goals": sum(g["pool_recoverable"].casefold() == "true" and g["goal_catalog_id"] in covered_ids for g in goals),
                "selected_candidates": candidate_total, "merged_delivery_segments": segment_total,
                "unique_delivery_minutes_total": round(minutes_total, 3),
                "unique_delivery_minutes_per_match": round(minutes_total / len(matches), 3),
                "reviewed_no_goal_controls_selected": sum(any(r["group_uid"] == uid for r in selected_by_method_cap[(method, cap, group_by_uid[uid]["match_id"])]) for uid in negative_uids),
            })

    pair_rows = []
    for control in controls:
        goal_uid, control_uid = control["missed_goal_group_uid"], control["control_group_uid"]
        for method in METHODS:
            pos_score = predictions[method][goal_uid]
            neg_score = predictions[method][control_uid]
            pair_rows.append({
                "method": method, "goal_catalog_id": control["goal_catalog_id"], "match_id": control["match_id"],
                "goal_group_uid": goal_uid, "goal_rank_baseline": control["missed_goal_rank_full_pool"],
                "control_group_uid": control_uid, "control_rank_baseline": control["control_rank_full_pool"],
                "positive_score": pos_score, "reviewed_no_goal_score": neg_score,
                "goal_ranked_above_control": pos_score > neg_score,
            })

    pair_summary = []
    for method in METHODS:
        local = [row for row in pair_rows if row["method"] == method]
        wins = sum(bool(row["goal_ranked_above_control"]) for row in local)
        pair_summary.append({"method": method, "matched_pairs": len(local), "goals_above_reviewed_no_goal": wins,
                             "pairwise_accuracy": round(wins / len(local), 4)})
    baseline_k60 = next(row for row in summary_rows if row["method"] == "RF_BASELINE_FROZEN" and row["k"] == 60)
    expected_baseline = (1470, 1126, 1415.824, 128)
    actual_baseline = (int(baseline_k60["selected_candidates"]), int(baseline_k60["merged_delivery_segments"]),
                       round(float(baseline_k60["unique_delivery_minutes_total"]), 3), int(baseline_k60["covered_catalog_goals"]))
    if actual_baseline != expected_baseline:
        raise ValueError(f"Seleção baseline divergente: obtido={actual_baseline}, esperado={expected_baseline}")

    args.output.mkdir(parents=True)
    write_csv(args.output / "reviewed_hard_negative_lomo_scores.csv", score_rows)
    write_csv(args.output / "reviewed_hard_negative_pairwise_lomo.csv", pair_rows)
    write_csv(args.output / "reviewed_hard_negative_rerank_summary.csv", summary_rows)
    write_csv(args.output / "reviewed_hard_negative_rerank_by_match.csv", per_match_rows)

    md = [
        "# Reranking com hard negatives revisados — TRAIN LOMO",
        "",
        "## Dados e protocolo",
        "",
        f"Positives: {len(positive_uids)} recovered catalog groups. Negatives: {len(negative_uids)} human-confirmed NO_GOAL groups; other weak labels are excluded. LOMO is by match.",
        f"Progressive batch review: {batch_review_counts} (only the new confirmed NO_GOAL rows were added to the prior confirmed controls).",
        "",
        "Fixed models: frozen RF baseline; LR C=0.1 with 36/49 features; RF 300 trees, leaf=2, balanced on 49 features; XGBoost 300 estimators, depth=3, learning_rate=0.05, subsample/colsample=0.8. No hyperparameter search.",
        "",
        "Selection uses gap45 and delivery windows 20+40s at K=20/40/60. Recall covers catalogued goal timestamps; unique minutes are delivered video duration.",
        "",
        "## Separação dos pares difíceis revisados",
        "",
        "| Método | Pares | Gol acima do controle NO_GOAL | Acurácia pareada |",
        "|---|---:|---:|---:|",
    ]
    for row in pair_summary:
        md.append(f"| {row['method']} | {row['matched_pairs']} | {row['goals_above_reviewed_no_goal']} | {row['pairwise_accuracy']:.1%} |")
    md.extend([
        "",
        "## Cobertura e custo por K",
        "",
        "| Método | K | Gols cobertos | Recall | Minutos únicos/jogo | Segmentos | Controles NO_GOAL selecionados |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        md.append(f"| {row['method']} | {row['k']} | {row['covered_catalog_goals']}/{row['catalog_goals']} | {float(row['catalog_recall']):.1%} | {row['unique_delivery_minutes_per_match']:.3f} | {row['merged_delivery_segments']} | {row['reviewed_no_goal_controls_selected']} |")
    md.extend([
        "",
        "## Limites",
        "",
        "Exploratory TRAIN-only reranking. Negative coverage is a reviewed sample, not all false candidates; no winner is promoted without an independent batch. VALIDATION and TEST stay closed.",
        "",
        "O `inside_goal_fraction` atual representa a fração de hits dentro de uma caixa expandida da goleira, não tempo físico dentro da boca do gol. As features de 49 dimensões acrescentam tendências em bins temporais, mas também derivam das detecções existentes; não resolvem identidade errada da bola.",
        "",
        f"Outputs: `{args.output}`; reviewed controls: `{args.controls}`.",
    ])
    (args.output / "reviewed_hard_negative_reranking.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (args.output / "reviewed_hard_negative_reranking.json").write_text(json.dumps({
        "scope": "TRAIN only; LOMO by match; no VALIDATION/TEST; weak_label=0 excluded from negative training labels",
        "methods": list(METHODS),
        "positive_groups": len(positive_uids), "manually_confirmed_no_goal_groups": len(negative_uids),
        "prior_control_count": len(prior_controls), "progressive_batch_review": batch_review_counts,
        "features_base": len(base_features), "features_rich": len(rich_features),
        "baseline_gap45_k60_reproduced": actual_baseline,
        "pairwise_summary": pair_summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"positive_groups": len(positive_uids), "reviewed_no_goal_groups": len(negative_uids),
                      "baseline": actual_baseline, "pairwise": pair_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
