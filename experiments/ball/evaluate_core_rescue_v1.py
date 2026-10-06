"""Compare fixed-budget CORE+RESCUE selections with the temporal ranking on TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from build_complementarity_matrix_v1 import (
    BASE_SCORES,
    EXPERTS,
    GAP_SECONDS,
    LIMITS,
    ROOT,
    RRF_OFFSET,
    covered_goals,
    merged_cost,
    read_csv,
    select_with_gap,
    write_csv,
)


DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/CORE-RESCUE-EXP-0001"
OUTPUT_FILES = {
    "core_rescue_comparison_v1.csv", "core_rescue_selected_candidates_v1.csv",
    "core_rescue_by_match_v1.csv", "core_rescue_goal_coverage_v1.csv", "summary_v1.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report87-k-grid", action="store_true",
                        help="Also evaluate fixed-budget K=20 and K=80 variants for the Report 87 grid.")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        existing = {path.name for path in args.output.iterdir() if path.is_file()}
        summary_path = args.output / "summary_v1.json"
        if existing - OUTPUT_FILES or not summary_path.is_file():
            raise FileExistsError(f"Saída contém artefatos não reconhecidos; não vou sobrescrever: {args.output}")
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("experiment_id") != args.output.name:
            raise FileExistsError(f"A saída existente não pertence a este experimento: {args.output}")

    pool = read_csv(args.candidate_scores)
    pool_by_key = {(row["match_id"], row["group_uid"]): row for row in pool}
    experts: dict[str, dict[tuple[str, str], float]] = {}
    for name, path in EXPERTS.items():
        experts[name] = {
            (row["match_id"], row["group_uid"]): float(row["rank_score"])
            for row in read_csv(path)
        }
    common = set(pool_by_key)
    for scores in experts.values():
        common &= set(scores)
    if len(common) != 5372:
        raise ValueError(f"Esperados 5.372 grupos TRAIN comuns, encontrados {len(common)}")

    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(common):
        row: dict[str, object] = dict(pool_by_key[key])
        for name, scores in experts.items():
            row[name] = scores[key]
        row["_key"] = f"{key[0]}|{key[1]}"
        by_match[key[0]].append(row)

    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Somente TRAIN pode ser lido nesta análise.")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or sum(map(len, goals.values())) != 130 or set(by_match) != set(goals):
        raise ValueError("A população precisa coincidir com o protocolo frozen de 22 partidas/130 gols.")

    methods: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    for match, rows in by_match.items():
        for score in [*BASE_SCORES, *EXPERTS]:
            methods[score][match] = sorted(rows, key=lambda r: (-float(r[score]), str(r["group_uid"])))
        rrf_sources = [*BASE_SCORES, "net_motion"]
        reciprocal: dict[str, float] = defaultdict(float)
        for source in rrf_sources:
            for rank, row in enumerate(methods[source][match], start=1):
                reciprocal[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
        methods["RRF4_plus_net_motion"][match] = sorted(
            rows, key=lambda r: (-reciprocal[str(r["group_uid"])], str(r["group_uid"]))
        )

    # CORE size, ordered rescue sources, and total budget per match. The 35+10+10
    # variant follows the two-source example in report 87.
    variants = [
        {"name": "CORE20_RESCUE10_NET", "core": 20, "rescues": [("net_motion", 10)]},
        {"name": "CORE30_RESCUE10_NET", "core": 30, "rescues": [("net_motion", 10)]},
        {"name": "CORE30_RESCUE20_NET", "core": 30, "rescues": [("net_motion", 20)]},
        {"name": "CORE35_RESCUE10_FUSION_RESCUE10_NET", "core": 35, "rescues": [("fusion_score_goal", 10), ("net_motion", 10)]},
        {"name": "CORE40_RESCUE10_NET", "core": 40, "rescues": [("net_motion", 10)]},
        {"name": "CORE40_RESCUE20_NET", "core": 40, "rescues": [("net_motion", 20)]},
        {"name": "CORE40_RESCUE20_RRF_NET", "core": 40, "rescues": [("RRF4_plus_net_motion", 20)]},
    ]
    if args.report87_k_grid:
        variants.extend([
            {"name": "CORE10_RESCUE10_NET", "core": 10, "rescues": [("net_motion", 10)]},
            {"name": "CORE60_RESCUE10_NET_RESCUE10_FUSION", "core": 60,
             "rescues": [("net_motion", 10), ("fusion_score_goal", 10)]},
        ])

    selections: dict[str, dict[str, list[dict[str, object]]]] = {}
    baseline_selections: dict[str, dict[str, list[dict[str, object]]]] = {}
    spec_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for variant in variants:
        name = str(variant["name"])
        core_k = int(variant["core"])
        rescue_specs = list(variant["rescues"])
        total_k = core_k + sum(int(count) for _, count in rescue_specs)
        per_match: dict[str, list[dict[str, object]]] = {}
        baseline_per_match: dict[str, list[dict[str, object]]] = {}
        for match in by_match:
            core = [
                {**row, "_selection_stage": "CORE", "_selection_source": "rank_score"}
                for row in select_with_gap(methods["rank_score"][match], core_k)
            ]
            chosen = list(core)
            chosen_keys = {str(row["_key"]) for row in chosen}
            rescue_counts: dict[str, int] = defaultdict(int)
            for source, count in rescue_specs:
                for row in methods[source][match]:
                    key = str(row["_key"])
                    center = float(row["delivery_start_seconds"]) + 20.0
                    if key in chosen_keys:
                        continue
                    if any(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) < GAP_SECONDS for other in chosen):
                        continue
                    chosen.append({**row, "_selection_stage": "RESCUE", "_selection_source": source})
                    chosen_keys.add(key)
                    rescue_counts[source] += 1
                    if rescue_counts[source] >= int(count):
                        break
            baseline_available = select_with_gap(methods["rank_score"][match], total_k)
            if args.report87_k_grid:
                effective_k = min(len(chosen), len(baseline_available))
                chosen = chosen[:effective_k]
                baseline_per_match[match] = baseline_available[:effective_k]
            else:
                if len(chosen) != total_k:
                    raise ValueError(f"{name}/{match}: obtidos {len(chosen)} de {total_k} candidatos planejados")
                if len(baseline_available) != len(chosen):
                    raise ValueError(f"{name}/{match}: baseline tem {len(baseline_available)} itens para {len(chosen)} híbridos")
                baseline_per_match[match] = baseline_available
            per_match[match] = chosen
            for row in chosen:
                candidate_rows.append({
                    "variant": name, "match_id": match, "group_uid": row["group_uid"],
                    "stage": row["_selection_stage"],
                    "source": row["_selection_source"],
                    "candidate_start_seconds": row["candidate_start_seconds"],
                    "candidate_end_seconds": row["candidate_end_seconds"],
                    "delivery_start_seconds": row["delivery_start_seconds"],
                    "delivery_end_seconds": row["delivery_end_seconds"],
                    "rank_score": row["rank_score"],
                    "fusion_score_goal": row["fusion_score_goal"],
                    "net_motion": row["net_motion"],
                })
        selections[name] = per_match
        baseline_selections[name] = baseline_per_match

        baseline = baseline_per_match
        selected_goals = covered_goals(per_match, goals)
        baseline_goals = covered_goals(baseline, goals)
        segments, minutes = merged_cost(per_match)
        baseline_segments, base_minutes = merged_cost(baseline)
        spec_rows.append({
            "variant": name, "core_k": core_k,
            "rescue_spec": ";".join(f"{source}:{count}" for source, count in rescue_specs),
            "budget_k_per_match": total_k, "covered_goals": len(selected_goals),
            "effective_candidates_per_match_min": min(map(len, per_match.values())),
            "effective_candidates_per_match_max": max(map(len, per_match.values())),
            "baseline_temporal_same_budget_goals": len(baseline_goals),
            "net_goal_change_vs_baseline": len(selected_goals) - len(baseline_goals),
            "goals_rescued_vs_baseline": len(selected_goals - baseline_goals),
            "baseline_goals_lost": len(baseline_goals - selected_goals),
            "selected_candidates": sum(map(len, per_match.values())),
            "merged_segments": segments, "baseline_merged_segments": baseline_segments,
            "unique_review_minutes_per_match": f"{minutes:.3f}",
            "baseline_unique_review_minutes_per_match": f"{base_minutes:.3f}",
        })

    per_match_rows: list[dict[str, object]] = []
    goal_rows: list[dict[str, object]] = []
    for variant in variants:
        name = str(variant["name"])
        total_k = int(variant["core"]) + sum(int(n) for _, n in variant["rescues"])
        baseline = baseline_selections[name]
        selected_goals = covered_goals(selections[name], goals)
        baseline_goals = covered_goals(baseline, goals)
        for match, goal_list in goals.items():
            chosen_intervals = [(float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in selections[name][match]]
            base_intervals = [(float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in baseline[match]]
            won = lost = 0
            for goal in goal_list:
                timestamp = float(goal["event_timestamp_seconds"])
                hybrid_hit = any(start <= timestamp <= end for start, end in chosen_intervals)
                baseline_hit = any(start <= timestamp <= end for start, end in base_intervals)
                goal_rows.append({"variant": name, "k_per_match": total_k,
                                  "effective_k_for_match": len(selections[name][match]),
                                  "goal_catalog_id": goal["goal_catalog_id"], "match_id": match,
                                  "hybrid_covered": hybrid_hit, "baseline_covered": baseline_hit})
                won += int(hybrid_hit and not baseline_hit)
                lost += int(baseline_hit and not hybrid_hit)
            per_match_rows.append({"variant": name, "match_id": match, "k_per_match": total_k,
                                   "effective_candidates": len(selections[name][match]),
                                   "hybrid_goals_rescued": won, "baseline_goals_lost": lost,
                                   "net_change": won-lost})

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "core_rescue_comparison_v1.csv", spec_rows)
    write_csv(args.output / "core_rescue_selected_candidates_v1.csv", candidate_rows)
    write_csv(args.output / "core_rescue_by_match_v1.csv", per_match_rows)
    write_csv(args.output / "core_rescue_goal_coverage_v1.csv", goal_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matches; 5,372 common candidate groups; 130 completed catalog goals; all input scores OOF by match",
        "protocol": {"core_method": "temporal rank_score", "gap_seconds": GAP_SECONDS,
                     "delivery_window": "20+40 seconds",
                     "budget": ("requested K per match; in the Report 87 grid, realized candidate count is capped to the smaller achievable count of hybrid and temporal Top-K under the 45s gap, matched per match"
                                if args.report87_k_grid else "fixed per match; compared with temporal Top-K at equal total K"),
                     "rescue_rule": "source rank order; exclude candidates already selected and candidates within the frozen 45s gap",
                     "rescue_order": "fixed as listed in each variant; no fitted weights", "limits_per_match": sorted({int(v["core"])+sum(int(n) for _,n in v["rescues"]) for v in variants})},
        "variants": variants,
        "warning": "Exploratory development result on the common TRAIN/LOMO subset; not an operating-policy promotion. No VALIDATION or TEST data accessed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": args.output.name, "variants": len(variants), "scope": summary["scope"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
