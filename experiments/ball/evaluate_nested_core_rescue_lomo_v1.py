"""Leakage-controlled outer LOMO for the frozen CORE+RESCUE policy grid on TRAIN."""
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
DEFAULT_OUT = ROOT / "experiments/ball/NESTED-CORE-RESCUE-LOMO-EXP-0001"
GRID = (
    ("CORE30_FUSION10_NET10", 30, "fusion", 10, "net_motion", 10),
    ("CORE35_FUSION10_NET10", 35, "fusion", 10, "net_motion", 10),
    ("CORE40_FUSION10_NET10", 40, "fusion", 10, "net_motion", 10),
    ("CORE40_RRF3_10_NET10", 40, "rrf3", 10, "net_motion", 10),
    ("CORE40_RRF3_10_NET20", 40, "rrf3", 10, "net_motion", 20),
    ("CORE40_NET20", 40, "net_motion", 20, "", 0),
)


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
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)


def lr() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))


def rank_map(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (-float(row[field]), str(row["group_uid"])))
    return {str(row["group_uid"]): rank for rank, row in enumerate(ordered, 1)}


def selections(rows: list[dict[str, object]], policy: tuple[str, int, str, int, str, int]) -> list[dict[str, object]]:
    _, core_k, first, first_k, second, second_k = policy
    ranked: dict[str, list[dict[str, object]]] = {}
    for source in ("rank_score", "visual_score_goal", "fusion_score_goal", "net_motion"):
        ranked[source] = sorted(rows, key=lambda row: (-float(row[source]), str(row["group_uid"])))
    ranked["fusion"] = ranked["fusion_score_goal"]
    rankmaps = {source: {str(row["group_uid"]): i for i, row in enumerate(items, 1)} for source, items in ranked.items()}
    rrf_scores: dict[str, float] = defaultdict(float)
    for source in ("rank_score", "visual_score_goal", "fusion_score_goal"):
        for uid, rank in rankmaps[source].items():
            rrf_scores[uid] += 1.0 / (60 + rank)
    ranked["rrf3"] = sorted(rows, key=lambda row: (-rrf_scores[str(row["group_uid"])], str(row["group_uid"])))
    ranked["rrf3"] = [{**row, "_rrf3": rrf_scores[str(row["group_uid"])]} for row in ranked["rrf3"]]
    chosen: list[dict[str, object]] = []

    def add(source: str, limit: int) -> None:
        added = 0
        for row in ranked[source]:
            center = float(row["delivery_start_seconds"]) + 20.0
            if any(str(row["group_uid"]) == str(other["group_uid"]) or abs(center - (float(other["delivery_start_seconds"]) + 20.0)) < 45.0 for other in chosen):
                continue
            chosen.append(row)
            added += 1
            if added >= limit:
                break

    add("rank_score", core_k)
    if first:
        add(first, first_k)
    if second:
        add(second, second_k)
    return chosen


def score_policy(rows_by_match: dict[str, list[dict[str, object]]], goals: dict[str, list[dict[str, str]]], policy: tuple[str, int, str, int, str, int]) -> dict[str, object]:
    selected = {match: selections(rows, policy) for match, rows in rows_by_match.items()}
    covered = 0
    goal_total = 0
    segments = 0
    seconds = 0.0
    candidates = 0
    for match, picks in selected.items():
        intervals = [(float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in picks]
        covered += sum(any(start <= float(goal["event_timestamp_seconds"]) <= end for start, end in intervals) for goal in goals[match])
        goal_total += len(goals[match]); candidates += len(picks)
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segments += len(merged); seconds += sum(end - start for start, end in merged)
    return {"policy": policy[0], "covered": covered, "goals": goal_total, "candidates": candidates,
            "segments": segments, "unique_minutes": seconds / 60.0 / max(1, len(selected))}


def policy_key(result: dict[str, object], complexity: int) -> tuple[float, float, float, int]:
    return (float(result["covered"]), -float(result["unique_minutes"]), -float(result["segments"]), -complexity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.n_estimators != 300:
        raise ValueError("The registered A1 protocol freezes n_estimators=300")

    temporal_path = ROOT / "results/local_trajectory_dataset_train_fullpool_v1/temporal_group_dataset_local_trajectory_v1.csv"
    net_path = ROOT / "results/net_motion_dataset_train_v1/temporal_group_dataset_net_motion_v1.csv"
    candidate_path = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1/candidate_scores_lomo_v1.csv"
    candidate_embeddings_path = ROOT / "experiments/ball/GOAL-EVENT-FULL-POOL-VISUAL-RERANKER-EXP-0001/run_v1_embeddings_checkpoint.npy"
    candidate_embedding_manifest = candidate_embeddings_path.with_suffix(".json")
    verifier_path = ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/predictions_lomo_v1.csv"
    verifier_embeddings_path = ROOT / "experiments/ball/GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001/run_v1/sequence_embeddings_v1.npz"
    goals_path = ROOT / "results/local_trajectory_dataset_train_v2/goal_timestamps_train_subset_v1.csv"
    frozen_ranker_path = ROOT / "experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib"
    input_paths = [temporal_path, net_path, candidate_path, candidate_embeddings_path, candidate_embedding_manifest, verifier_path, verifier_embeddings_path, goals_path, frozen_ranker_path]
    if any(not path.is_file() for path in input_paths):
        raise FileNotFoundError([str(path) for path in input_paths if not path.is_file()])

    temporal, temporal_fields = read_csv(temporal_path)
    net, net_fields = read_csv(net_path)
    candidate_rows, candidate_fields = read_csv(candidate_path)
    verifier_rows, _ = read_csv(verifier_path)
    goals_rows, goal_fields = read_csv(goals_path)
    if any(row.get("split", "train").lower() != "train" for row in goals_rows):
        raise AssertionError("A1 accepts TRAIN catalog goals only")
    frozen = joblib.load(frozen_ranker_path)
    temporal_features = list(frozen["features"])
    excluded = {"group_uid", "match_id", "best_timestamp_seconds", "weak_label"}
    net_features = [field for field in net_fields if field not in excluded]
    temporal_by_key = {(r["match_id"], r["group_uid"]): r for r in temporal}
    net_by_key = {(r["match_id"], r["group_uid"]): r for r in net}
    candidates_by_key = {(r["match_id"], r["group_uid"]): r for r in candidate_rows}
    common = set(temporal_by_key) & set(net_by_key) & set(candidates_by_key)
    if len(common) != 5372:
        raise ValueError(f"Expected the frozen common population of 5,372 groups; found {len(common)}")

    manifest = json.loads(candidate_embedding_manifest.read_text(encoding="utf-8"))
    candidate_emb = np.load(candidate_embeddings_path, mmap_mode="r", allow_pickle=False)
    candidate_id_to_index = {candidate_id: i for i, candidate_id in enumerate(manifest["candidate_ids"])}
    if len(candidate_id_to_index) != len(candidate_rows) or candidate_emb.shape != (len(candidate_rows), 4096):
        raise ValueError("Candidate embedding manifest, score table and tensor do not align")
    cemb_by_key: dict[tuple[str, str], np.ndarray] = {}
    for key in common:
        cid = candidates_by_key[key]["candidate_id"]
        cemb_by_key[key] = candidate_emb[candidate_id_to_index[cid]]

    verifier_archive = np.load(verifier_embeddings_path, allow_pickle=False)
    verifier_by_id = {str(cid): emb.reshape(-1) for cid, emb in zip(verifier_archive["clip_ids"], verifier_archive["embeddings"])}
    verifier_x = np.stack([verifier_by_id[row["clip_id"]] for row in verifier_rows])
    verifier_y = np.asarray([int(row["label"] == "GOAL") for row in verifier_rows], dtype=np.int32)
    verifier_matches = np.asarray([row["match_id"] for row in verifier_rows])
    if len(verifier_rows) != 304 or set(verifier_y) != {0, 1}:
        raise ValueError("Unexpected TRAIN verifier examples")

    match_ids = sorted({key[0] for key in common})
    if len(match_ids) != 22:
        raise ValueError(f"Expected 22 TRAIN outer folds, found {len(match_ids)}")
    temporal_x = np.asarray([[float(temporal_by_key[key][f]) for f in temporal_features] for key in sorted(common)], dtype=np.float32)
    net_x = np.asarray([[float(net_by_key[key][f]) for f in net_features] for key in sorted(common)], dtype=np.float32)
    keys = sorted(common)
    key_index = {key: i for i, key in enumerate(keys)}
    temporal_y = np.asarray([int(temporal_by_key[key]["weak_label"]) for key in keys], dtype=np.int32)
    net_y = np.asarray([int(net_by_key[key]["weak_label"]) for key in keys], dtype=np.int32)
    if not np.array_equal(temporal_y, net_y):
        raise ValueError("Temporal/net-motion labels differ on the frozen common pool")
    group_rows: dict[tuple[str, str], dict[str, object]] = {}
    for key in keys:
        source = candidates_by_key[key]
        group_rows[key] = {**source, "group_uid": key[1], "match_id": key[0], "catalog_goal_in_clip": int(source["catalog_goal_in_clip"])}
    goals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in goals_rows:
        if row["annotation_status"] == "completed" and row.get("event_timestamp_seconds"):
            goals[row["match_id"]].append(row)
    if sum(map(len, goals.values())) != 130 or set(goals) != set(match_ids):
        raise ValueError("A1 expected exactly the frozen set of 130 TRAIN catalog goals over all 22 matches")

    outer_predictions: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for outer_index, outer_match in enumerate(match_ids, 1):
        dev_matches = [m for m in match_ids if m != outer_match]
        dev_base: dict[tuple[str, str], dict[str, object]] = {}
        inner_visual: dict[str, np.ndarray] = {}
        inner_temporal: dict[str, np.ndarray] = {}
        inner_net: dict[str, np.ndarray] = {}
        for inner_match in dev_matches:
            train_match_mask = np.asarray([k[0] not in {outer_match, inner_match} for k in keys])
            held_keys = [key for key in keys if key[0] == inner_match]
            held_indices = np.asarray([key_index[key] for key in held_keys], dtype=int)
            tm = rf().fit(temporal_x[train_match_mask], temporal_y[train_match_mask])
            nm = rf().fit(net_x[train_match_mask], net_y[train_match_mask])
            inner_temporal[inner_match] = tm.predict_proba(temporal_x[held_indices])[:, list(tm.classes_).index(1)]
            inner_net[inner_match] = nm.predict_proba(net_x[held_indices])[:, list(nm.classes_).index(1)]
            vm_mask = (verifier_matches != outer_match) & (verifier_matches != inner_match)
            vm = lr().fit(verifier_x[vm_mask], verifier_y[vm_mask])
            candidate_embeddings = np.stack([cemb_by_key[key] for key in held_keys])
            inner_visual[inner_match] = vm.predict_proba(candidate_embeddings)[:, 1]
            for idx, key in enumerate(held_keys):
                dev_base[key] = {**group_rows[key], "rank_score": float(inner_temporal[inner_match][idx]),
                                 "visual_score_goal": float(inner_visual[inner_match][idx]), "net_motion": float(inner_net[inner_match][idx])}
        dev_base_rows = list(dev_base.values())
        # Produce inner-held fusion scores: fit only on other development matches' cross-fitted base scores.
        dev_fusion: dict[str, np.ndarray] = {}
        for inner_match in dev_matches:
            fit_rows = [r for r in dev_base_rows if r["match_id"] != inner_match]
            test_rows = [r for r in dev_base_rows if r["match_id"] == inner_match]
            fx = np.asarray([[float(r["rank_score"]), float(r["visual_score_goal"])] for r in fit_rows], dtype=np.float64)
            fy = np.asarray([int(r["catalog_goal_in_clip"]) for r in fit_rows], dtype=np.int32)
            fm = lr().fit(fx, fy)
            tx = np.asarray([[float(r["rank_score"]), float(r["visual_score_goal"])] for r in test_rows], dtype=np.float64)
            dev_fusion[inner_match] = fm.predict_proba(tx)[:, 1]
            for row, score in zip(test_rows, dev_fusion[inner_match]):
                row["fusion_score_goal"] = float(score)

        inner_by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in dev_base_rows:
            inner_by_match[str(row["match_id"])].append(row)
        policy_results: list[tuple[tuple[str, int, str, int, str, int], dict[str, object]]] = []
        for complexity, policy in enumerate(GRID):
            result = score_policy(inner_by_match, {m: goals[m] for m in dev_matches}, policy)
            selection_rows.append({"outer_match": outer_match, **result, "policy_complexity_order": complexity})
            policy_results.append((policy, result))
        chosen_policy, chosen_inner = max(policy_results, key=lambda item: policy_key(item[1], GRID.index(item[0])))

        # Fit each base source on the complete outer development fold, then score only the held-out match.
        dev_mask = np.asarray([k[0] != outer_match for k in keys])
        held_keys = [key for key in keys if key[0] == outer_match]
        held_indices = np.asarray([key_index[key] for key in held_keys], dtype=int)
        tm = rf().fit(temporal_x[dev_mask], temporal_y[dev_mask])
        nm = rf().fit(net_x[dev_mask], net_y[dev_mask])
        outer_t = tm.predict_proba(temporal_x[held_indices])[:, list(tm.classes_).index(1)]
        outer_n = nm.predict_proba(net_x[held_indices])[:, list(nm.classes_).index(1)]
        vm_mask = verifier_matches != outer_match
        vm = lr().fit(verifier_x[vm_mask], verifier_y[vm_mask])
        outer_emb = np.stack([cemb_by_key[key] for key in held_keys])
        outer_v = vm.predict_proba(outer_emb)[:, 1]
        fusion_fit = dev_base_rows
        fx = np.asarray([[float(r["rank_score"]), float(r["visual_score_goal"])] for r in fusion_fit], dtype=np.float64)
        fy = np.asarray([int(r["catalog_goal_in_clip"]) for r in fusion_fit], dtype=np.int32)
        fm = lr().fit(fx, fy)
        outer_f = fm.predict_proba(np.column_stack([outer_t, outer_v]))[:, 1]
        held_rows: list[dict[str, object]] = []
        for idx, key in enumerate(held_keys):
            held_rows.append({**group_rows[key], "rank_score": float(outer_t[idx]), "visual_score_goal": float(outer_v[idx]),
                              "fusion_score_goal": float(outer_f[idx]), "net_motion": float(outer_n[idx]), "outer_match": outer_match})
        selected = selections(held_rows, chosen_policy)
        intervals = [(float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in selected]
        covered_goals = [g for g in goals[outer_match] if any(a <= float(g["event_timestamp_seconds"]) <= b for a, b in intervals)]
        merged: list[list[float]] = []
        for a, b in sorted(intervals):
            if not merged or a > merged[-1][1]: merged.append([a, b])
            else: merged[-1][1] = max(merged[-1][1], b)
        minutes = sum(b-a for a,b in merged) / 60.0
        fold_rows.append({"outer_match": outer_match, "policy_selected": chosen_policy[0], "inner_covered_goals": chosen_inner["covered"],
                          "inner_dev_goals": chosen_inner["goals"], "outer_goals": len(goals[outer_match]), "outer_covered_goals": len(covered_goals),
                          "outer_selected_candidates": len(selected), "outer_merged_segments": len(merged), "outer_unique_review_minutes": f"{minutes:.3f}"})
        for goal in goals[outer_match]:
            outer_predictions.append({"match_id": outer_match, "goal_catalog_id": goal["goal_catalog_id"],
                                      "goal_timestamp_seconds": goal["event_timestamp_seconds"],
                                      "covered": any(float(r["delivery_start_seconds"]) <= float(goal["event_timestamp_seconds"]) <= float(r["delivery_end_seconds"]) for r in selected),
                                      "policy_selected": chosen_policy[0]})
        if outer_index % args.progress_every == 0 or outer_index == len(match_ids):
            print(f"outer_fold={outer_index}/{len(match_ids)} held_match={outer_match} policy={chosen_policy[0]} covered={len(covered_goals)}/{len(goals[outer_match])}", flush=True)

    total_goals = sum(int(r["outer_goals"]) for r in fold_rows)
    total_covered = sum(int(r["outer_covered_goals"]) for r in fold_rows)
    summary = {"experiment_id": args.output.name, "scope": "TRAIN-only nested LOMO; every outer match excluded from all inner training, policy selection, and fusion fitting; TEST/VALIDATION not accessed",
               "matches": len(fold_rows), "catalog_goals": total_goals, "covered_goals": total_covered,
               "recall": f"{total_covered/total_goals:.6f}", "mean_unique_review_minutes": f"{sum(float(r['outer_unique_review_minutes']) for r in fold_rows)/len(fold_rows):.3f}",
               "models": "Temporal RandomForest and net_motion RandomForest (300 trees, min_samples_leaf=2, balanced, random_state=42); visual LogisticRegression on frozen ResNet18 embeddings; fusion LogisticRegression on cross-fitted temporal+visual scores",
               "grid": [p[0] for p in GRID], "selection_rule": "maximize aggregate covered catalog goals, then minimize unique video-minutes, then segments, then use lower grid complexity order",
               "warning": "Outer LOMO estimates this frozen policy-selection procedure on TRAIN development data; not independent validation and not a promotion decision.",
               "inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}}
    args.output.mkdir(parents=True)
    write_csv(args.output / "nested_lomo_outer_folds_v1.csv", fold_rows)
    write_csv(args.output / "nested_lomo_policy_selection_v1.csv", selection_rows)
    write_csv(args.output / "nested_lomo_goal_coverage_v1.csv", outer_predictions)
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
