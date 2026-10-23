"""Leakage-controlled 25-match TRAIN-only nested LOMO for fixed review budgets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1"
WINDOWS = {
    30: (10.0, 20.0),
    40: (40.0 / 3.0, 80.0 / 3.0),
    45: (15.0, 30.0),
    60: (20.0, 40.0),
}
BUDGETS = (30, 40, 45, 50, 58)
K_VALUES = (20, 30, 40, 50, 55, 60)
GAP_SECONDS = 45.0
METHODS = (
    "TEMPORAL",
    "RRF3",
    "RRF4_NET",
    "CORE35_FUSION10_NET10",
    "CORE40_RRF4_NET_RESCUE",
)
POLICY_FAMILIES = {
    "GLOBAL_BEST": METHODS,
    "NO_RESCUE": ("TEMPORAL", "RRF3", "RRF4_NET"),
    "CORE_RESCUE": ("CORE35_FUSION10_NET10", "CORE40_RRF4_NET_RESCUE"),
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
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


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


def lr() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
    )


def add_separated(
    chosen: list[dict[str, object]], ranked: list[dict[str, object]], limit: int,
) -> None:
    added = 0
    for row in ranked:
        center = float(row["delivery_start_seconds"]) + 20.0
        if any(
            str(row["group_uid"]) == str(other["group_uid"])
            or abs(center - (float(other["delivery_start_seconds"]) + 20.0)) < GAP_SECONDS
            for other in chosen
        ):
            continue
        chosen.append(row)
        added += 1
        if added >= limit:
            break


def rrf_rank(rows: list[dict[str, object]], sources: tuple[str, ...]) -> list[dict[str, object]]:
    scores: dict[str, float] = defaultdict(float)
    for source in sources:
        ordered = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
        for rank, row in enumerate(ordered, 1):
            scores[str(row["group_uid"])] += 1.0 / (60 + rank)
    return sorted(rows, key=lambda row: (-scores[str(row["group_uid"])], str(row["group_uid"])))


def select_policy(rows: list[dict[str, object]], method: str, k: int) -> list[dict[str, object]]:
    rrf3 = rrf_rank(rows, ("rank_score", "visual_score_goal", "fusion_score_goal"))
    rrf4 = rrf_rank(rows, ("rank_score", "visual_score_goal", "fusion_score_goal", "net_motion"))
    ranked = {
        "TEMPORAL": sorted(rows, key=lambda row: (-float(row["rank_score"]), str(row["group_uid"]))),
        "RRF3": rrf3,
        "RRF4_NET": rrf4,
        "FUSION": sorted(rows, key=lambda row: (-float(row["fusion_score_goal"]), str(row["group_uid"]))),
        "NET": sorted(rows, key=lambda row: (-float(row["net_motion"]), str(row["group_uid"]))),
    }
    if method in {"TEMPORAL", "RRF3", "RRF4_NET"}:
        chosen: list[dict[str, object]] = []
        add_separated(chosen, ranked[method], k)
        return chosen

    if method == "CORE35_FUSION10_NET10":
        core_size, rescue_sources = 35, (("FUSION", 10), ("NET", 10))
    elif method == "CORE40_RRF4_NET_RESCUE":
        core_size, rescue_sources = 40, (("RRF4_NET", 20),)
    else:
        raise ValueError(f"Unknown policy: {method}")

    chosen = []
    add_separated(chosen, ranked["TEMPORAL"], core_size)
    for source, quota in rescue_sources:
        if len(chosen) >= k:
            break
        add_separated(chosen, ranked[source], min(quota, k - len(chosen)))
    return chosen[:k]


def delivery_rows(rows: list[dict[str, object]], window: int) -> list[dict[str, object]]:
    pre, post = WINDOWS[window]
    changed: list[dict[str, object]] = []
    for row in rows:
        center = float(row["delivery_start_seconds"]) + 20.0
        changed.append({
            **row,
            "delivery_start_seconds": max(0.0, center - pre),
            "delivery_end_seconds": center + post,
        })
    return changed


def cost_minutes(rows: list[dict[str, object]]) -> tuple[int, float]:
    intervals = sorted(
        (float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"]))
        for row in rows
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return len(merged), sum(end - start for start, end in merged) / 60.0


def score_policy(
    rows_by_match: dict[str, list[dict[str, object]]],
    goals: dict[str, list[dict[str, str]]],
    method: str,
    k: int,
    window: int,
) -> dict[str, object]:
    total_covered = 0
    total_goals = 0
    segments = 0
    candidates = 0
    per_match_minutes: list[float] = []
    for match, source_rows in rows_by_match.items():
        candidates_for_window = delivery_rows(source_rows, window)
        selected = select_policy(candidates_for_window, method, k)
        intervals = [
            (float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"]))
            for row in selected
        ]
        total_covered += sum(
            any(start <= float(goal["event_timestamp_seconds"]) <= end for start, end in intervals)
            for goal in goals[match]
        )
        total_goals += len(goals[match])
        count, minutes = cost_minutes(selected)
        segments += count
        candidates += len(selected)
        per_match_minutes.append(minutes)
    mean_minutes = sum(per_match_minutes) / max(1, len(per_match_minutes))
    return {
        "method": method,
        "k_per_match": k,
        "window_seconds": window,
        "covered_goals": total_covered,
        "catalog_goals": total_goals,
        "recall": total_covered / max(1, total_goals),
        "selected_candidates": candidates,
        "merged_segments": segments,
        "mean_unique_minutes_per_match": mean_minutes,
        "max_unique_minutes_per_match": max(per_match_minutes, default=0.0),
        "budget_feasible_all_matches": True,
    }


def complexity_order(method: str, window: int, k: int) -> tuple[int, int, int]:
    return (METHODS.index(method), window, k)


def policy_key(result: dict[str, object], method: str, window: int, k: int) -> tuple[float, float, float, tuple[int, int, int]]:
    return (
        float(result["covered_goals"]),
        -float(result["mean_unique_minutes_per_match"]),
        -float(result["merged_segments"]),
        tuple(-x for x in complexity_order(method, window, k)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {args.output}")
    if args.n_estimators != 300:
        raise ValueError("The nested time-budget protocol freezes n_estimators=300")

    temporal_path = ROOT / "results/local_trajectory_dataset_train_fullpool_v1/temporal_group_dataset_local_trajectory_v1.csv"
    net_path = ROOT / "results/net_motion_dataset_train_fullpool_nested_v1/temporal_group_dataset_net_motion_v1.csv"
    candidate_path = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
    candidate_embeddings_path = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.npy"
    candidate_embedding_manifest = candidate_embeddings_path.with_suffix(".json")
    verifier_path = ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/predictions_lomo_v1.csv"
    verifier_embeddings_path = ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/sequence_embeddings_v1.npz"
    goals_path = ROOT / "results/local_trajectory_dataset_train_fullpool_v1/goal_timestamps_train_subset_v1.csv"
    frozen_ranker_path = ROOT / "experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib"
    input_paths = [
        temporal_path, net_path, candidate_path, candidate_embeddings_path,
        candidate_embedding_manifest, verifier_path, verifier_embeddings_path,
        goals_path, frozen_ranker_path,
    ]
    if any(not path.is_file() for path in input_paths):
        raise FileNotFoundError([str(path) for path in input_paths if not path.is_file()])

    temporal, temporal_fields = read_csv(temporal_path)
    net, net_fields = read_csv(net_path)
    candidate_rows, _ = read_csv(candidate_path)
    verifier_rows, _ = read_csv(verifier_path)
    goals_rows, _ = read_csv(goals_path)
    if any(row.get("split", "train").lower() != "train" for row in goals_rows):
        raise AssertionError("Nested time-budget LOMO accepts TRAIN catalog goals only")

    frozen = joblib.load(frozen_ranker_path)
    temporal_features = list(frozen["features"])
    excluded = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    net_features = [field for field in net_fields if field not in excluded]
    temporal_by_key = {(row["match_id"], row["group_uid"]): row for row in temporal}
    net_by_key = {(row["match_id"], row["group_uid"]): row for row in net}
    candidates_by_key = {(row["match_id"], row["group_uid"]): row for row in candidate_rows}
    common = set(temporal_by_key) & set(net_by_key) & set(candidates_by_key)
    if len(common) != 5742:
        raise ValueError(f"Expected full-pool common population of 5,742 groups; found {len(common)}")

    embedding_manifest = json.loads(candidate_embedding_manifest.read_text(encoding="utf-8"))
    candidate_embeddings = np.load(candidate_embeddings_path, mmap_mode="r", allow_pickle=False)
    candidate_id_to_index = {candidate_id: i for i, candidate_id in enumerate(embedding_manifest["candidate_ids"])}
    if len(candidate_id_to_index) != len(candidate_rows) or candidate_embeddings.shape != (len(candidate_rows), 4096):
        raise ValueError("Candidate embedding manifest, score table and tensor do not align")
    candidate_embeddings_by_key = {
        key: candidate_embeddings[candidate_id_to_index[candidates_by_key[key]["candidate_id"]]]
        for key in common
    }

    verifier_archive = np.load(verifier_embeddings_path, allow_pickle=False)
    verifier_by_id = {
        str(clip_id): embedding.reshape(-1)
        for clip_id, embedding in zip(verifier_archive["clip_ids"], verifier_archive["embeddings"])
    }
    verifier_x = np.stack([verifier_by_id[row["clip_id"]] for row in verifier_rows])
    verifier_y = np.asarray([int(row["label"] == "GOAL") for row in verifier_rows], dtype=np.int32)
    verifier_matches = np.asarray([row["match_id"] for row in verifier_rows])
    if len(verifier_rows) != 304 or set(verifier_y) != {0, 1}:
        raise ValueError("Unexpected frozen TRAIN verifier examples")

    keys = sorted(common)
    match_ids = sorted({key[0] for key in keys})
    if len(match_ids) != 25:
        raise ValueError(f"Expected 25 TRAIN outer folds, found {len(match_ids)}")
    key_index = {key: index for index, key in enumerate(keys)}
    temporal_x = np.asarray(
        [[float(temporal_by_key[key][field]) for field in temporal_features] for key in keys], dtype=np.float32
    )
    net_x = np.asarray(
        [[float(net_by_key[key][field]) for field in net_features] for key in keys], dtype=np.float32
    )
    temporal_y = np.asarray([int(temporal_by_key[key]["weak_label"]) for key in keys], dtype=np.int32)
    net_y = np.asarray([int(net_by_key[key]["weak_label"]) for key in keys], dtype=np.int32)
    if not np.array_equal(temporal_y, net_y):
        raise ValueError("Temporal/net-motion labels differ on the frozen common pool")

    group_rows: dict[tuple[str, str], dict[str, object]] = {}
    for key in keys:
        candidate = candidates_by_key[key]
        group_rows[key] = {
            **candidate,
            "group_uid": key[1],
            "match_id": key[0],
            "catalog_goal_in_clip": int(candidate["catalog_goal_in_clip"]),
        }

    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in goals_rows:
        if row["annotation_status"] == "completed" and row.get("event_timestamp_seconds"):
            goals[row["match_id"]].append(row)
    if sum(map(len, goals.values())) != 152 or set(goals) != set(match_ids):
        raise ValueError("Expected exactly 152 completed TRAIN goals over 25 matches")

    fold_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    goal_rows: list[dict[str, object]] = []
    variants = [(method, window, k) for method in METHODS for window in WINDOWS for k in K_VALUES]

    for outer_index, outer_match in enumerate(match_ids, 1):
        dev_matches = [match for match in match_ids if match != outer_match]
        dev_base: dict[tuple[str, str], dict[str, object]] = {}
        for inner_match in dev_matches:
            train_mask = np.asarray([key[0] not in {outer_match, inner_match} for key in keys])
            held_keys = [key for key in keys if key[0] == inner_match]
            held_indices = np.asarray([key_index[key] for key in held_keys], dtype=int)
            temporal_model = rf().fit(temporal_x[train_mask], temporal_y[train_mask])
            net_model = rf().fit(net_x[train_mask], net_y[train_mask])
            temporal_scores = temporal_model.predict_proba(temporal_x[held_indices])[:, list(temporal_model.classes_).index(1)]
            net_scores = net_model.predict_proba(net_x[held_indices])[:, list(net_model.classes_).index(1)]
            verifier_mask = (verifier_matches != outer_match) & (verifier_matches != inner_match)
            verifier_model = lr().fit(verifier_x[verifier_mask], verifier_y[verifier_mask])
            inner_embeddings = np.stack([candidate_embeddings_by_key[key] for key in held_keys])
            visual_scores = verifier_model.predict_proba(inner_embeddings)[:, 1]
            for index, key in enumerate(held_keys):
                dev_base[key] = {
                    **group_rows[key],
                    "rank_score": float(temporal_scores[index]),
                    "visual_score_goal": float(visual_scores[index]),
                    "net_motion": float(net_scores[index]),
                }

        dev_base_rows = list(dev_base.values())
        for inner_match in dev_matches:
            fit_rows = [row for row in dev_base_rows if row["match_id"] != inner_match]
            held_rows = [row for row in dev_base_rows if row["match_id"] == inner_match]
            fusion_x = np.asarray(
                [[float(row["rank_score"]), float(row["visual_score_goal"])] for row in fit_rows], dtype=np.float64
            )
            fusion_y = np.asarray([int(row["catalog_goal_in_clip"]) for row in fit_rows], dtype=np.int32)
            fusion_model = lr().fit(fusion_x, fusion_y)
            held_x = np.asarray(
                [[float(row["rank_score"]), float(row["visual_score_goal"])] for row in held_rows], dtype=np.float64
            )
            for row, score in zip(held_rows, fusion_model.predict_proba(held_x)[:, 1]):
                row["fusion_score_goal"] = float(score)

        inner_by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in dev_base_rows:
            inner_by_match[str(row["match_id"])].append(row)

        scores_by_variant: dict[tuple[str, int, int], dict[str, object]] = {}
        for method, window, k in variants:
            result = score_policy(inner_by_match, {match: goals[match] for match in dev_matches}, method, k, window)
            scores_by_variant[(method, window, k)] = result
        selected_by_budget: dict[int, dict[str, tuple[tuple[str, int, int], dict[str, object]]]] = {}
        for budget in BUDGETS:
            for method, window, k in variants:
                result = scores_by_variant[(method, window, k)]
                is_feasible = float(result["max_unique_minutes_per_match"]) <= budget + 1e-9
                selection_rows.append({
                    "outer_match": outer_match,
                    "budget_minutes_per_match": budget,
                    **result,
                    "budget_feasible_all_inner_matches": is_feasible,
                })
            selected_by_budget[budget] = {}
            for family, family_methods in POLICY_FAMILIES.items():
                feasible = [
                    ((method, window, k), scores_by_variant[(method, window, k)])
                    for method in family_methods
                    for window in WINDOWS
                    for k in K_VALUES
                    if float(scores_by_variant[(method, window, k)]["max_unique_minutes_per_match"]) <= budget + 1e-9
                ]
                if not feasible:
                    raise RuntimeError(
                        f"No {family} policy respects the strict {budget}-minute cap in all inner matches for outer={outer_match}"
                    )
                selected_by_budget[budget][family] = max(
                    feasible,
                    key=lambda item: policy_key(item[1], item[0][0], item[0][1], item[0][2]),
                )

        # Refit base rankers on the complete outer development fold; held-out data
        # participates only in prediction and the final outer-fold measurement.
        outer_train_mask = np.asarray([key[0] != outer_match for key in keys])
        held_keys = [key for key in keys if key[0] == outer_match]
        held_indices = np.asarray([key_index[key] for key in held_keys], dtype=int)
        temporal_model = rf().fit(temporal_x[outer_train_mask], temporal_y[outer_train_mask])
        net_model = rf().fit(net_x[outer_train_mask], net_y[outer_train_mask])
        outer_temporal = temporal_model.predict_proba(temporal_x[held_indices])[:, list(temporal_model.classes_).index(1)]
        outer_net = net_model.predict_proba(net_x[held_indices])[:, list(net_model.classes_).index(1)]
        verifier_mask = verifier_matches != outer_match
        verifier_model = lr().fit(verifier_x[verifier_mask], verifier_y[verifier_mask])
        outer_embeddings = np.stack([candidate_embeddings_by_key[key] for key in held_keys])
        outer_visual = verifier_model.predict_proba(outer_embeddings)[:, 1]
        fusion_fit = dev_base_rows
        fusion_x = np.asarray(
            [[float(row["rank_score"]), float(row["visual_score_goal"])] for row in fusion_fit], dtype=np.float64
        )
        fusion_y = np.asarray([int(row["catalog_goal_in_clip"]) for row in fusion_fit], dtype=np.int32)
        fusion_model = lr().fit(fusion_x, fusion_y)
        outer_fusion = fusion_model.predict_proba(np.column_stack([outer_temporal, outer_visual]))[:, 1]
        outer_rows: list[dict[str, object]] = []
        for index, key in enumerate(held_keys):
            outer_rows.append({
                **group_rows[key],
                "rank_score": float(outer_temporal[index]),
                "visual_score_goal": float(outer_visual[index]),
                "fusion_score_goal": float(outer_fusion[index]),
                "net_motion": float(outer_net[index]),
                "outer_match": outer_match,
            })

        for budget, family_choices in selected_by_budget.items():
            for family, (variant, inner_result) in family_choices.items():
                method, window, k = variant
                held_window_rows = delivery_rows(outer_rows, window)
                selected = select_policy(held_window_rows, method, k)
                segment_count, held_minutes = cost_minutes(selected)
                intervals = [
                    (float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"]))
                    for row in selected
                ]
                covered = 0
                for goal in goals[outer_match]:
                    timestamp = float(goal["event_timestamp_seconds"])
                    hit = any(start <= timestamp <= end for start, end in intervals)
                    covered += int(hit)
                    goal_rows.append({
                        "budget_minutes_per_match": budget,
                        "policy_family": family,
                        "match_id": outer_match,
                        "goal_catalog_id": goal["goal_catalog_id"],
                        "goal_timestamp_seconds": goal["event_timestamp_seconds"],
                        "covered": hit,
                        "policy_selected": method,
                        "window_seconds": window,
                        "k_per_match": k,
                    })
                fold_rows.append({
                    "budget_minutes_per_match": budget,
                    "policy_family": family,
                    "outer_match": outer_match,
                    "policy_selected": method,
                    "window_seconds": window,
                    "k_per_match": k,
                    "inner_covered_goals": inner_result["covered_goals"],
                    "inner_catalog_goals": inner_result["catalog_goals"],
                    "inner_mean_unique_minutes": f"{float(inner_result['mean_unique_minutes_per_match']):.3f}",
                    "inner_max_unique_minutes": f"{float(inner_result['max_unique_minutes_per_match']):.3f}",
                    "outer_goals": len(goals[outer_match]),
                    "outer_covered_goals": covered,
                    "outer_selected_candidates": len(selected),
                    "outer_merged_segments": segment_count,
                    "outer_unique_review_minutes": f"{held_minutes:.3f}",
                    "outer_budget_exceeded": held_minutes > budget + 1e-9,
                })

        if outer_index % args.progress_every == 0 or outer_index == len(match_ids):
            chosen = ", ".join(
                f"{budget}:{family}={policy[0][0]}/{policy[0][1]}s/K{policy[0][2]}"
                for budget, families in sorted(selected_by_budget.items())
                for family, policy in families.items()
            )
            print(f"outer_fold={outer_index}/{len(match_ids)} held_match={outer_match} policies={chosen}", flush=True)

    summary_rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        for family in POLICY_FAMILIES:
            rows = [
                row for row in fold_rows
                if int(row["budget_minutes_per_match"]) == budget and row["policy_family"] == family
            ]
            total_goals = sum(int(row["outer_goals"]) for row in rows)
            total_covered = sum(int(row["outer_covered_goals"]) for row in rows)
            policy_counts: dict[str, int] = {}
            for row in rows:
                policy = f"{row['policy_selected']}|W{row['window_seconds']}|K{row['k_per_match']}"
                policy_counts[policy] = policy_counts.get(policy, 0) + 1
            summary_rows.append({
                "budget_minutes_per_match": budget,
                "policy_family": family,
                "covered_goals": total_covered,
                "catalog_goals": total_goals,
                "recall": f"{total_covered / max(1, total_goals):.6f}",
                "mean_unique_minutes_per_match": f"{sum(float(row['outer_unique_review_minutes']) for row in rows) / len(rows):.3f}",
                "max_outer_match_minutes": f"{max(float(row['outer_unique_review_minutes']) for row in rows):.3f}",
                "budget_exceeded_outer_matches": sum(bool(row["outer_budget_exceeded"]) for row in rows),
                "mean_merged_segments_per_match": f"{sum(int(row['outer_merged_segments']) for row in rows) / len(rows):.3f}",
                "policy_frequency_json": json.dumps(policy_counts, ensure_ascii=False, sort_keys=True),
            })

    baseline_by_budget = {
        int(row["budget_minutes_per_match"]): row
        for row in summary_rows if row["policy_family"] == "NO_RESCUE"
    }
    by_budget_match = {
        (int(row["budget_minutes_per_match"]), str(row["outer_match"])): row
        for row in fold_rows if row["policy_family"] == "NO_RESCUE"
    }
    for summary_row in summary_rows:
        budget = int(summary_row["budget_minutes_per_match"])
        family = str(summary_row["policy_family"])
        family_rows = [
            row for row in fold_rows
            if int(row["budget_minutes_per_match"]) == budget and row["policy_family"] == family
        ]
        comparisons = [
            (row, by_budget_match[(budget, str(row["outer_match"]))])
            for row in family_rows
        ]
        summary_row["paired_delta_goals_vs_no_rescue"] = sum(
            int(row["outer_covered_goals"]) - int(baseline["outer_covered_goals"])
            for row, baseline in comparisons
        )
        summary_row["matches_gain_vs_no_rescue"] = sum(
            int(row["outer_covered_goals"]) > int(baseline["outer_covered_goals"])
            for row, baseline in comparisons
        )
        summary_row["matches_loss_vs_no_rescue"] = sum(
            int(row["outer_covered_goals"]) < int(baseline["outer_covered_goals"])
            for row, baseline in comparisons
        )
        summary_row["matches_tied_vs_no_rescue"] = sum(
            int(row["outer_covered_goals"]) == int(baseline["outer_covered_goals"])
            for row, baseline in comparisons
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "nested_time_budget_summary_v1.csv", summary_rows)
    write_csv(args.output / "nested_time_budget_outer_folds_v1.csv", fold_rows)
    write_csv(args.output / "nested_time_budget_policy_selection_v1.csv", selection_rows)
    write_csv(args.output / "nested_time_budget_goal_coverage_v1.csv", goal_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN-only nested LOMO over all 25 evaluable matches; outer match excluded from model fitting, inner policy selection and evaluation; VALIDATION and TEST not accessed",
        "matches": len(match_ids),
        "catalog_goals": sum(map(len, goals.values())),
        "common_candidate_groups": len(common),
        "outer_folds": len(fold_rows) // (len(BUDGETS) * len(POLICY_FAMILIES)),
        "budgets_minutes_per_match": list(BUDGETS),
        "budget_constraint": "hard cap: every inner development match must have merged unique video-minutes <= budget; held-out outer match compliance is measured and reported",
        "grid": {
            "methods": list(METHODS),
            "k_per_match": list(K_VALUES),
            "windows_seconds": list(WINDOWS),
            "gap_seconds": GAP_SECONDS,
            "window_center": "frozen candidate center, reconstructed from source delivery_start_seconds + 20s",
        },
        "policy_families": {name: list(methods) for name, methods in POLICY_FAMILIES.items()},
        "paired_comparison": "Global best and CORE_RESCUE are compared with a separately nested NO_RESCUE selector at each identical per-match budget and held-out outer match.",
        "selection_rule": "among policies feasible on every inner match, maximize aggregate covered catalog goals, then minimize mean unique minutes, merged segments, and fixed grid complexity order",
        "models": "Temporal/net-motion RandomForest (300 trees, min_samples_leaf=2, balanced, random_state=42); visual LogisticRegression on frozen ResNet18 candidate embeddings; fusion LogisticRegression on cross-fitted temporal+visual scores",
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "script_sha256": sha256(Path(__file__).resolve()),
        "interpretation": "TRAIN nested LOMO estimate only; not independent VALIDATION, not a final mode selection and not permission to open TEST",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary_rows, "output": str(args.output.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
