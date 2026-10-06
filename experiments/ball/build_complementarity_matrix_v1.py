"""Build pairwise method and catalog-goal complementarity tables on TRAIN/LOMO."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
DEFAULT_GOALS = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
EXPERTS = {
    "local_trajectory": ROOT / "results/local_trajectory_lomo_train_v2/lomo_predictions_v1.csv",
    "shot_motion": ROOT / "results/shot_motion_lomo_train_v1/lomo_predictions_v1.csv",
    "net_motion": ROOT / "results/net_motion_lomo_train_v1/lomo_predictions_v1.csv",
}
BASE_SCORES = ("rank_score", "visual_score_goal", "fusion_score_goal")
LIMITS = (20, 40, 60, 80, 100)
GAP_SECONDS = 45.0
RRF_OFFSET = 60


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Sem linhas para gravar: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_with_gap(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        center = float(row["delivery_start_seconds"]) + 20.0
        if all(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) >= GAP_SECONDS for other in selected):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def covered_goals(selected: dict[str, list[dict[str, object]]], goals: dict[str, list[dict[str, object]]]) -> set[str]:
    covered: set[str] = set()
    for match, goal_rows in goals.items():
        intervals = [
            (float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"]))
            for row in selected.get(match, [])
        ]
        for goal in goal_rows:
            timestamp = float(goal["event_timestamp_seconds"])
            if any(start <= timestamp <= end for start, end in intervals):
                covered.add(goal["goal_catalog_id"])
    return covered


def merged_cost(rows_by_match: dict[str, list[dict[str, object]]]) -> tuple[int, float]:
    segments = 0
    review_seconds = 0.0
    for rows in rows_by_match.values():
        intervals = sorted((float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in rows)
        merged: list[list[float]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segments += len(merged)
        review_seconds += sum(end - start for start, end in merged)
    return segments, review_seconds / 60.0 / max(1, len(rows_by_match))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--goal-timestamps", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/ball/COMPLEMENTARITY-MATRIX-EXP-0001")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Saída já contém arquivos: {args.output}")

    pool = read_csv(args.candidate_scores)
    required = {"match_id", "group_uid", "candidate_id", "delivery_start_seconds", "delivery_end_seconds", *BASE_SCORES}
    if not pool or not required <= set(pool[0]):
        raise ValueError(f"Pool sem colunas obrigatórias: {sorted(required - set(pool[0] if pool else []))}")
    pool_by_key = {(row["match_id"], row["group_uid"]): row for row in pool}
    if len(pool_by_key) != len(pool):
        raise ValueError("Chave match_id/group_uid duplicada no pool base.")

    expert_maps: dict[str, dict[tuple[str, str], float]] = {}
    for name, path in EXPERTS.items():
        rows = read_csv(path)
        if not rows or not {"match_id", "group_uid", "rank_score"} <= set(rows[0]):
            raise ValueError(f"Arquivo de score sem schema esperado: {path}")
        expert_maps[name] = {(row["match_id"], row["group_uid"]): float(row["rank_score"]) for row in rows}
    common_keys = set(pool_by_key)
    for scores in expert_maps.values():
        common_keys &= set(scores)
    if len(common_keys) != 5372:
        raise ValueError(f"A interseção LOMO esperada era 5.372 grupos; encontrada {len(common_keys)}")

    candidates: list[dict[str, object]] = []
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    score_names = [*BASE_SCORES, *EXPERTS]
    for key in sorted(common_keys):
        source = pool_by_key[key]
        row: dict[str, object] = {**source}
        for name, scores in expert_maps.items():
            row[name] = scores[key]
        row["_key"] = f"{key[0]}|{key[1]}"
        candidates.append(row)
        by_match[key[0]].append(row)

    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in read_csv(args.goal_timestamps):
        if goal.get("split", "train").lower() != "train":
            raise AssertionError("Somente TRAIN é permitido; VALIDATION/TEST não podem entrar nesta análise.")
        if goal.get("annotation_status") == "completed" and goal.get("event_timestamp_seconds") and goal["match_id"] in by_match:
            goals[goal["match_id"]].append(goal)
    if len(by_match) != 22 or sum(map(len, goals.values())) != 130 or set(by_match) != set(goals):
        raise ValueError(f"Escopo esperado 22 partidas/130 gols; obtido {len(by_match)}/{sum(map(len, goals.values()))}")

    rrf_sources = {
        "RRF3_temporal_visual_fusion": list(BASE_SCORES),
        "RRF4_plus_local_trajectory": [*BASE_SCORES, "local_trajectory"],
        "RRF4_plus_shot_motion": [*BASE_SCORES, "shot_motion"],
        "RRF4_plus_net_motion": [*BASE_SCORES, "net_motion"],
        "RRF6_plus_all_three": [*BASE_SCORES, *EXPERTS],
    }
    methods = list(score_names) + list(rrf_sources)
    ranked: dict[str, dict[str, list[dict[str, object]]]] = {name: {} for name in methods}
    for match, rows in by_match.items():
        individual_rankings = {
            score: sorted(rows, key=lambda r: (-float(r[score]), str(r["group_uid"])))
            for score in score_names
        }
        for score, ranking in individual_rankings.items():
            ranked[score][match] = ranking
        for method, sources in rrf_sources.items():
            reciprocal: dict[str, float] = defaultdict(float)
            for source in sources:
                for rank, row in enumerate(individual_rankings[source], start=1):
                    reciprocal[str(row["group_uid"])] += 1.0 / (RRF_OFFSET + rank)
            ranked[method][match] = sorted(rows, key=lambda r: (-reciprocal[str(r["group_uid"])], str(r["group_uid"])))

    selected: dict[tuple[str, int], dict[str, list[dict[str, object]]]] = {}
    candidate_sets: dict[tuple[str, int], set[str]] = {}
    goal_sets: dict[tuple[str, int], set[str]] = {}
    method_rows: list[dict[str, object]] = []
    goal_rows: list[dict[str, object]] = []
    for method in methods:
        for k in LIMITS:
            per_match = {match: select_with_gap(ranked[method][match], k) for match in by_match}
            selected[(method, k)] = per_match
            candidate_sets[(method, k)] = {
                str(row["_key"]) for rows in per_match.values() for row in rows
            }
            goal_sets[(method, k)] = covered_goals(per_match, goals)
            segments, minutes = merged_cost(per_match)
            method_rows.append({
                "method": method, "k_per_match": k, "catalog_goals_evaluable": 130,
                "covered_goals": len(goal_sets[(method, k)]),
                "selected_candidates": len(candidate_sets[(method, k)]),
                "merged_segments": segments,
                "unique_review_minutes_per_match": f"{minutes:.3f}",
            })
            for match, goal_list in goals.items():
                intervals = [(float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in per_match[match]]
                for goal in goal_list:
                    timestamp = float(goal["event_timestamp_seconds"])
                    goal_rows.append({
                        "method": method, "k_per_match": k,
                        "goal_catalog_id": goal["goal_catalog_id"], "match_id": match,
                        "covered": any(start <= timestamp <= end for start, end in intervals),
                    })

    pair_rows: list[dict[str, object]] = []
    for k in LIMITS:
        for i, left in enumerate(methods):
            for right in methods[i + 1:]:
                c_left, c_right = candidate_sets[(left, k)], candidate_sets[(right, k)]
                g_left, g_right = goal_sets[(left, k)], goal_sets[(right, k)]
                c_union, g_union = c_left | c_right, g_left | g_right
                union_rows_by_match: dict[str, list[dict[str, object]]] = {}
                left_rows = selected[(left, k)]
                right_rows = selected[(right, k)]
                for match in by_match:
                    dedup: dict[str, dict[str, object]] = {}
                    for row in [*left_rows[match], *right_rows[match]]:
                        dedup[str(row["_key"])] = row
                    union_rows_by_match[match] = list(dedup.values())
                segments, minutes = merged_cost(union_rows_by_match)
                pair_rows.append({
                    "k_per_match": k, "method_a": left, "method_b": right,
                    "candidate_intersection": len(c_left & c_right),
                    "candidate_union": len(c_union),
                    "candidate_jaccard": f"{len(c_left & c_right) / len(c_union):.6f}" if c_union else "1.000000",
                    "goal_intersection": len(g_left & g_right),
                    "goal_union_covered": len(g_union),
                    "goal_jaccard": f"{len(g_left & g_right) / len(g_union):.6f}" if g_union else "1.000000",
                    "goals_only_a": len(g_left - g_right), "goals_only_b": len(g_right - g_left),
                    "union_candidate_count": len(c_union),
                    "union_merged_segments": segments,
                    "union_unique_review_minutes_per_match": f"{minutes:.3f}",
                })

    args.output.mkdir(parents=True)
    write_csv(args.output / "method_coverage_v1.csv", method_rows)
    write_csv(args.output / "pairwise_complementarity_v1.csv", pair_rows)
    write_csv(args.output / "goal_coverage_by_method_v1.csv", goal_rows)
    source_paths = [args.candidate_scores, args.goal_timestamps, *EXPERTS.values()]
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 22 matches; 5,372 common candidate groups; 130 completed catalog goals; OOF scores by match",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths],
        "excluded": {"candidate_groups": len(pool) - len(common_keys), "reason": "expert LOMO scores missing for 3 matches/370 groups"},
        "methods": methods,
        "protocol": {
            "delivery_window": "20+40 seconds as encoded in candidate input",
            "temporal_dedup_gap_seconds": GAP_SECONDS,
            "limits_per_match": list(LIMITS),
            "rrf_offset": RRF_OFFSET,
            "goal_coverage": "catalog timestamp lies within selected delivery interval",
            "candidate_overlap": "Jaccard over selected (match_id, group_uid) sets",
            "goal_overlap": "Jaccard over covered catalog goal IDs",
            "selection": "candidate sets and goal labels are descriptive; no weights fitted or configuration promoted",
        },
        "warning": "Exploratory development comparison on TRAIN/LOMO. Results are limited to the common 22-match/130-goal population; no VALIDATION or TEST data accessed.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": args.output.name, "methods": len(methods), "pair_rows": len(pair_rows), "goal_rows": len(goal_rows), "scope": summary["scope"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
