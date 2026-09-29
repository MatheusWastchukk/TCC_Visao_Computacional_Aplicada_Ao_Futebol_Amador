"""Compara fontes LOMO de trajetória/bola no RRF, somente em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_AUDIT = ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/manual_review_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/GOAL-EVENT-BALL-EVIDENCE-RRF-ENSEMBLE-EXP-0001"
EXPERTS = {
    "local_trajectory": ROOT / "results/local_trajectory_lomo_train_v2/lomo_predictions_v1.csv",
    "shot_motion": ROOT / "results/shot_motion_lomo_train_v1/lomo_predictions_v1.csv",
    "net_motion": ROOT / "results/net_motion_lomo_train_v1/lomo_predictions_v1.csv",
}
BASE_SCORES = ("rank_score", "visual_score_goal", "fusion_score_goal")
EXPERIMENTAL_SCORES = tuple(EXPERTS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para gravar em {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gap_select(rows: list[dict[str, object]], limit: int, gap: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        center = float(row["delivery_start_seconds"]) + 20.0
        if all(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) >= gap for other in selected):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def measure(method: str, k: int, selected: dict[str, list[dict[str, object]]], goals: dict[str, list[float]]) -> dict[str, object]:
    covered = sum(
        any(float(row["delivery_start_seconds"]) <= timestamp <= float(row["delivery_end_seconds"])
            for row in selected[match])
        for match, timestamps in goals.items() for timestamp in timestamps
    )
    segment_count = 0
    review_seconds = 0.0
    for rows in selected.values():
        merged: list[list[float]] = []
        for start, end in sorted((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in rows):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segment_count += len(merged)
        review_seconds += sum(end - start for start, end in merged)
    total_goals = sum(map(len, goals.values()))
    return {
        "method": method, "k_per_match": k, "catalog_goals": total_goals,
        "covered_goals": covered, "recall": f"{covered / total_goals:.6f}",
        "selected_candidates": sum(map(len, selected.values())), "merged_segments": segment_count,
        "unique_review_minutes_per_match": f"{review_seconds / 60 / len(selected):.3f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--manual-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gap-seconds", type=float, default=45.0)
    parser.add_argument("--limits", type=int, nargs="+", default=[20, 40, 60, 100])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}")
    pool = read_csv(args.candidate_scores)
    if not pool or not set(BASE_SCORES + ("match_id", "group_uid", "candidate_id", "delivery_start_seconds", "delivery_end_seconds")) <= set(pool[0]):
        raise ValueError("CSV base não contém scores, IDs de grupo e janelas necessários.")
    pool_by_key = {(row["match_id"], row["group_uid"]): row for row in pool}
    expert_maps: dict[str, dict[tuple[str, str], float]] = {}
    for name, path in EXPERTS.items():
        rows = read_csv(path)
        required = {"match_id", "group_uid", "rank_score"}
        if not rows or not required <= set(rows[0]):
            raise ValueError(f"Predições {name} sem colunas requeridas.")
        expert_maps[name] = {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in rows}
    common_keys = set(pool_by_key)
    for scores in expert_maps.values():
        common_keys &= set(scores)
    if len(common_keys) != 5372:
        raise ValueError(f"A interseção esperada de 5.372 grupos mudou: {len(common_keys)}")
    candidate_rows: list[dict[str, object]] = []
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    score_names = [*BASE_SCORES, *EXPERIMENTAL_SCORES]
    for key in sorted(common_keys):
        row = dict(pool_by_key[key])
        for name, scores in expert_maps.items():
            row[name] = scores[key]
        candidate_rows.append(row)
        by_match[row["match_id"]].append(row)

    goals: dict[str, list[float]] = defaultdict(list)
    goal_rows = read_csv(args.goal_timestamps)
    for row in goal_rows:
        if row.get("split", "train").lower() != "train":
            raise AssertionError("A avaliação aceita somente TRAIN; VALIDATION/TEST permanecem fechados.")
        if row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds") and row["match_id"] in by_match:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if len(by_match) != 22 or sum(map(len, goals.values())) != 130 or set(by_match) != set(goals):
        raise ValueError(f"Escopo esperado: 22 partidas e 130 gols avaliáveis; obtido {len(by_match)} partidas/{sum(map(len, goals.values()))} gols")

    standalone = ["rank_score", "visual_score_goal", "fusion_score_goal", *EXPERIMENTAL_SCORES]
    expert_sets = {
        "RRF3_temporal_visual_fusion": list(BASE_SCORES),
        "RRF4_plus_local_trajectory": [*BASE_SCORES, "local_trajectory"],
        "RRF4_plus_shot_motion": [*BASE_SCORES, "shot_motion"],
        "RRF4_plus_net_motion": [*BASE_SCORES, "net_motion"],
        "RRF6_plus_all_three": [*BASE_SCORES, *EXPERIMENTAL_SCORES],
    }
    rankings: dict[str, dict[str, list[dict[str, object]]]] = {}
    score_rankings: dict[str, dict[str, list[dict[str, object]]]] = {name: {} for name in standalone}
    for match, rows in by_match.items():
        for score_name in standalone:
            score_rankings[score_name][match] = sorted(rows, key=lambda row: (-float(row[score_name]), row["group_uid"]))
        for method, sources in expert_sets.items():
            reciprocal: dict[str, float] = defaultdict(float)
            for source in sources:
                for rank, row in enumerate(score_rankings[source][match], start=1):
                    reciprocal[str(row["group_uid"])] += 1.0 / (60.0 + rank)
            rankings.setdefault(method, {})[match] = sorted(
                rows, key=lambda row: (-reciprocal[str(row["group_uid"])], row["group_uid"])
            )

    result_rows: list[dict[str, object]] = []
    selected_by_method_k: dict[tuple[str, int], dict[str, list[dict[str, object]]]] = {}
    for name, per_match in score_rankings.items():
        for k in args.limits:
            selected = {match: gap_select(rows, k, args.gap_seconds) for match, rows in per_match.items()}
            result_rows.append(measure(f"standalone_{name}", k, selected, goals))
    for name, per_match in rankings.items():
        for k in args.limits:
            selected = {match: gap_select(rows, k, args.gap_seconds) for match, rows in per_match.items()}
            selected_by_method_k[(name, k)] = selected
            result_rows.append(measure(name, k, selected, goals))

    manual_rows = read_csv(args.manual_audit)
    manual_output: list[dict[str, object]] = []
    for audit in manual_rows:
        match = audit["match_id"]
        goal = float(audit["goal_timestamp_seconds"])
        for method, per_match in rankings.items():
            sorted_rows = per_match.get(match)
            if sorted_rows is None:
                for k in args.limits:
                    manual_output.append({
                        "goal_catalog_id": audit["goal_catalog_id"], "match_id": match,
                        "method": method, "k_per_match": k,
                        "representative_group_rank": "", "goal_covered": "not_evaluable_no_common_scores",
                    })
                continue
            rank_by_group = {str(row["group_uid"]): index for index, row in enumerate(sorted_rows, start=1)}
            group = audit.get("representative_group_uid", "")
            for k in args.limits:
                selected = selected_by_method_k[(method, k)][match]
                hit = any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in selected)
                manual_output.append({
                    "goal_catalog_id": audit["goal_catalog_id"], "match_id": match,
                    "method": method, "k_per_match": k,
                    "representative_group_rank": rank_by_group.get(group, ""),
                    "goal_covered": hit,
                })

    args.output.mkdir(parents=True)
    write_csv(args.output / "recall_at_k_comparison_v1.csv", result_rows)
    write_csv(args.output / "manual_miss_diagnostic_v1.csv", manual_output)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matched TRAIN games, 5,372 common groups, all rank scores LOMO; no VAL/TEST access",
        "catalog_goals": sum(map(len, goals.values())), "matches": len(by_match), "candidate_rows": len(candidate_rows),
        "excluded_from_feature_comparison": len(pool) - len(candidate_rows),
        "excluded_reason": "3 LOMO trajectory/shot/net score files do not contain these 370 groups/3 matches",
        "base_score_sources": list(BASE_SCORES), "tested_expert_sources": list(EXPERIMENTAL_SCORES),
        "selection": {"dedup_gap_seconds": args.gap_seconds, "limits_per_match": args.limits, "review_window": "delivery window 20+40s"},
        "label_protocol": "Existing completed TRAIN catalog timestamps; 130 goals; all rank predictions are out-of-fold by match.",
        "warning": "Exploratory RRF source comparison; no parameters fitted or promoted. Manual 12-miss labels are descriptive only and were not used to build ranks. TEST remains closed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
