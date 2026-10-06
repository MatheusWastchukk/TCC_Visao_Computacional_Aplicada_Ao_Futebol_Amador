"""Prepara a auditoria de ranking somente com artefatos TRAIN persistidos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
LIMITS = (20, 30, 40, 50, 60, 80)
CAUSES = (
    "B0_SEM_PROPOSTA_UTIL", "B1_CAIXA_IMPRECISA", "B2_MULTIPLICIDADE",
    "B3_CONTINUIDADE_VISUAL", "G1_GOLEIRA_SECUNDARIA", "G2_ASSOCIACAO_GOLEIRA",
    "R1_FALTA_SINAL_NAS_FEATURES", "R2_SINAL_PRESENTE_MAS_SCORE_BAIXO",
    "S1_SUPRESSAO", "S2_CORTE_TOPK", "V1_NAO_AVALIAVEL", "OUTRO",
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Recusado gravar tabela vazia: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_properties(path: Path) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not capture.isOpened() or fps <= 0 or frames <= 0:
            raise RuntimeError(f"NÃ£o foi possÃ­vel ler FPS/duraÃ§Ã£o: {path}")
        return fps, frames / fps
    finally:
        capture.release()


def clip_window(center: float, duration: float) -> tuple[float, float]:
    return max(0.0, center - 20.0), min(duration, center + 40.0)


def covers(row: dict[str, str], timestamp: float) -> bool:
    return float(row["delivery_start_seconds"]) <= timestamp <= float(row["delivery_end_seconds"])


def unique_minutes(rows: list[dict[str, str]]) -> tuple[float, int]:
    merged: list[list[float]] = []
    for row in sorted(rows, key=lambda item: float(item["delivery_start_seconds"])):
        start = float(row["delivery_start_seconds"])
        end = float(row["delivery_end_seconds"])
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged) / 60.0, len(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-predictions", type=Path, default=ROOT / "experiments/ball/GOAL-EVENT-TEMPORAL-RANKER-EXP-0002-predictions-v2/group_predictions_lomo_v1.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "experiments/ball/GOAL-EVENT-TEMPORAL-DATASET-V2/temporal_group_dataset_v1.csv")
    parser.add_argument("--shortlist", type=Path, default=ROOT / "experiments/ball/GOAL-EVENT-WIDE-POOL-TRAIN-V1/wide_pool_candidates_v1.csv")
    parser.add_argument("--score-audit", type=Path, default=ROOT / "experiments/ball/GOAL-EVENT-WIDE-POOL-OOF-REDUCER-TRAIN-V1/candidate_scores_lomo_v1.csv")
    parser.add_argument("--goal-catalog", type=Path, default=ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/train_goal_pool_and_shortlist_coverage_v1.csv")
    parser.add_argument("--manual-misses", type=Path, default=ROOT / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/manual_review_v1.csv")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001")
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--pair-comparators", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saÃ­da jÃ¡ existe; escolha uma nova versÃ£o: {args.output}")

    predictions, pred_fields = read_csv(args.rank_predictions)
    if "model" in pred_fields:
        predictions = [row for row in predictions if row.get("model") == "random_forest"]
        pred_fields = [field for field in pred_fields if field != "model"]
    features, feature_fields = read_csv(args.features)
    shortlist, short_fields = read_csv(args.shortlist)
    score_audit, score_fields = read_csv(args.score_audit)
    catalog, goal_fields = read_csv(args.goal_catalog)
    previous_misses, _ = read_csv(args.manual_misses)
    required_pred = {"group_uid", "match_id", "best_timestamp_seconds", "rank_score", "weak_label"}
    required_short = {"event_id", "match_id", "video_filename", "source_interval_id", "event_timestamp_seconds", "delivery_start_seconds", "delivery_end_seconds", "temporal_rank"}
    required_goal = {"goal_catalog_id", "match_id", "video_filename", "split", "event_timestamp_seconds", "is_pool_recoverable", "is_covered_by_rrf_nms_top60"}
    for label, rows, fields, required in (
        ("PrediÃ§Ãµes LOMO", predictions, pred_fields, required_pred),
        ("Shortlist", shortlist, short_fields, required_short),
        ("CatÃ¡logo TRAIN", catalog, goal_fields, required_goal),
    ):
        if not rows or (missing := required - set(fields)):
            raise ValueError(f"{label} vazio ou sem campos: {sorted(missing) if rows else 'vazio'}")
    if len(predictions) != 5742 or len({r["group_uid"] for r in predictions}) != len(predictions):
        raise ValueError("As prediÃ§Ãµes devem conter 5.742 grupos LOMO Ãºnicos de TRAIN.")
    if any(r.get("split", "train").lower() != "train" for r in catalog) or len(catalog) != 152:
        raise ValueError("O catÃ¡logo esperado tem 152 gols de TRAIN; VALIDATION/TEST sÃ£o proibidos.")
    if any(r.get("split", "train").lower() != "train" for r in shortlist):
        raise ValueError("A shortlist contÃ©m linha fora de TRAIN.")

    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for goal in catalog:
        if goal["annotation_status"] == "completed" and goal["event_timestamp_seconds"]:
            goals_by_match[goal["match_id"]].append(goal)
    predictions_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        predictions_by_match[row["match_id"]].append(row)
    shortlist_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in shortlist:
        shortlist_by_match[row["match_id"]].append(row)
    score_by_group = {(r["match_id"], r["group_uid"]): r for r in score_audit}
    feature_by_group = {(r["match_id"], r["group_uid"]): r for r in features}
    if len(score_by_group) != len(score_audit) or len(feature_by_group) != len(features):
        raise ValueError("IDs de grupos duplicados nos scores ou features.")
    matches = sorted(goals_by_match)
    if set(matches) != set(predictions_by_match) or set(matches) != set(shortlist_by_match):
        raise ValueError("Partidas de catÃ¡logo, scores LOMO e shortlist nÃ£o coincidem.")

    durations: dict[str, float] = {}
    source_fps: dict[str, float] = {}
    videos: dict[str, str] = {}
    for rows in goals_by_match.values():
        for row in rows:
            videos[row["match_id"]] = row["video_filename"]
    for match, filename in videos.items():
        source_fps[match], durations[match] = video_properties(args.videos_dir / filename)

    detector_sources: dict[str, str] = {}
    detector_source_sizes: dict[str, int] = {}
    miner_root = ROOT / "experiments/ball/GOAL-EVENT-CANDIDATE-MINER-V1"
    for path in miner_root.rglob("candidate_frames_v1.csv"):
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                first = next(csv.DictReader(stream), None)
            if first and first.get("match_id") in matches:
                match_id = first["match_id"]
                if path.stat().st_size > detector_source_sizes.get(match_id, -1):
                    detector_sources[match_id] = str(path.relative_to(ROOT))
                    detector_source_sizes[match_id] = path.stat().st_size
        except (OSError, StopIteration):
            continue

    ranked_pool: dict[str, list[dict[str, str]]] = {}
    rank_maps: dict[str, dict[str, int]] = {}
    for match, rows in predictions_by_match.items():
        ordered = sorted(rows, key=lambda r: (-float(r["rank_score"]), r["group_uid"]))
        ranked_pool[match] = ordered
        rank_maps[match] = {r["group_uid"]: i for i, r in enumerate(ordered, 1)}

    # Ensure the checked-in operational shortlist is exactly the existing RF
    # LOMO order with its fixed 45 s temporal gap, before using it as a baseline.
    for match, rows in ranked_pool.items():
        expected: list[dict[str, str]] = []
        for row in rows:
            center = float(row["best_timestamp_seconds"])
            if all(abs(center - float(other["best_timestamp_seconds"])) >= 45.0 for other in expected):
                expected.append(row)
            if len(expected) >= len(shortlist_by_match[match]):
                break
        expected_uids = [row["group_uid"] for row in expected]
        actual_uids = [row["source_interval_id"] for row in sorted(shortlist_by_match[match], key=lambda r: int(r["temporal_rank"]))]
        if expected_uids != actual_uids:
            raise ValueError(f"A shortlist operacional diverge do RF LOMO + gap 45 s em {match}.")

    # Rebuilds 20+40 delivery intervals using true video boundaries.
    pool_windows: dict[str, list[dict[str, str]]] = {}
    for match, rows in ranked_pool.items():
        windows = []
        for row in rows:
            start, end = clip_window(float(row["best_timestamp_seconds"]), durations[match])
            windows.append({**row, "video_filename": videos[match], "delivery_start_seconds": f"{start:.3f}", "delivery_end_seconds": f"{end:.3f}"})
        pool_windows[match] = windows
    pool_group = {(m, r["group_uid"]): r for m, rs in pool_windows.items() for r in rs}
    gap_selected_by_match: dict[str, list[dict[str, str]]] = {}
    gap_rank: dict[str, dict[str, int]] = {}
    for match, rows in ranked_pool.items():
        selected_rows: list[dict[str, str]] = []
        for row in rows:
            center = float(row["best_timestamp_seconds"])
            if all(abs(center - float(chosen["best_timestamp_seconds"])) >= 45.0 for chosen in selected_rows):
                selected_rows.append(pool_group[(match, row["group_uid"])])
            if len(selected_rows) >= max(LIMITS):
                break
        gap_selected_by_match[match] = selected_rows
        gap_rank[match] = {row["group_uid"]: index for index, row in enumerate(selected_rows, 1)}

    shortlist_sorted: dict[str, list[dict[str, str]]] = {}
    goal_rows: list[dict[str, object]] = []
    missed_goal_details: list[dict[str, object]] = []
    for match, goals in goals_by_match.items():
        short = []
        for source in sorted(shortlist_by_match[match], key=lambda r: int(r["temporal_rank"])):
            start = max(0.0, float(source["delivery_start_seconds"]))
            end = min(durations[match], float(source["delivery_end_seconds"]))
            short.append({**source, "delivery_start_seconds": f"{start:.3f}", "delivery_end_seconds": f"{end:.3f}"})
        shortlist_sorted[match] = short
        for goal in goals:
            timestamp = float(goal["event_timestamp_seconds"])
            recoveries = [r for r in pool_windows[match] if covers(r, timestamp)]
            recovered_best = min(recoveries, key=lambda r: rank_maps[match][r["group_uid"]]) if recoveries else None
            selected = [r for r in short if covers(r, timestamp)]
            pool_flag = bool(recoveries)
            top_flag = bool(selected)
            if str(goal["is_pool_recoverable"]).lower() != str(pool_flag).lower():
                raise ValueError(f"Teto do pool diverge do catÃ¡logo em {goal['goal_catalog_id']}; verificar janelas/fonte.")
            goal_rows.append({
                "goal_catalog_id": goal["goal_catalog_id"], "match_id": match,
                "video_filename": goal["video_filename"], "split": "train",
                "goal_timestamp_seconds": f"{timestamp:.3f}", "video_duration_seconds": f"{durations[match]:.3f}",
                "source_fps": f"{source_fps[match]:.6f}",
                "pool_recoverable": pool_flag, "best_temporal_group_uid": recovered_best["group_uid"] if recovered_best else "",
                "best_temporal_rank_full_pool": rank_maps[match][recovered_best["group_uid"]] if recovered_best else "",
                "best_temporal_score": recovered_best["rank_score"] if recovered_best else "",
                "covered_by_existing_top60": top_flag, "top60_group_uids": ";".join(r["source_interval_id"] for r in selected),
                "historical_rrf_nms_top60_coverage": goal["is_covered_by_rrf_nms_top60"],
                "historical_miss_audit_status": next((m.get("audit_status", "") for m in previous_misses if m.get("goal_catalog_id") == goal["goal_catalog_id"]), ""),
            })
            if pool_flag and not top_flag:
                missed_goal_details.append({"goal": goal, "best": recovered_best, "recoveries": recoveries})

    # Scores are complete only for the existing common top-60 shortlist.
    # RRF ranks below are diagnostic within that shortlist, not full-pool RRF.
    trace_rows: list[dict[str, object]] = []
    for match, rows in pool_windows.items():
        selected_by_uid = {r["source_interval_id"]: r for r in shortlist_sorted[match]}
        common = [score_by_group[(match, r["source_interval_id"])] for r in shortlist_sorted[match]
                  if (match, r["source_interval_id"]) in score_by_group]
        common_ranks: dict[str, dict[str, int]] = {}
        for field in ("rank_score", "visual_score_goal", "fusion_score_goal"):
            ordered = sorted(common, key=lambda r: (-float(r[field]), r["group_uid"]))
            common_ranks[field] = {r["group_uid"]: i for i, r in enumerate(ordered, 1)}
        reciprocal: dict[str, float] = defaultdict(float)
        for field in ("rank_score", "visual_score_goal", "fusion_score_goal"):
            for group_uid, rank in common_ranks[field].items():
                reciprocal[group_uid] += 1.0 / (60.0 + rank)
        rrf_order = sorted(reciprocal, key=lambda uid: (-reciprocal[uid], uid))
        rrf_rank = {uid: i for i, uid in enumerate(rrf_order, 1)}
        for row in rows:
            group = row["group_uid"]
            selected = selected_by_uid.get(group)
            scores = score_by_group.get((match, group), {})
            suppressor = next((candidate for candidate in shortlist_sorted[match]
                               if rank_maps[match][candidate["source_interval_id"]] < rank_maps[match][group]
                               and abs(float(candidate["event_timestamp_seconds"]) - float(row["best_timestamp_seconds"])) < 45.0), None)
            if selected:
                exclusion = "SELECTED_EXISTING_TOP60_GAP45"
            elif suppressor:
                exclusion = "GAP_45S_SUPPRESSED_BY_EARLIER_SELECTED_CANDIDATE"
            else:
                exclusion = "TOP60_CAPACITY_OR_AFTER_SHORTLIST_LIMIT"
            associated_goals = [g["goal_catalog_id"] for g in goals_by_match[match] if covers(row, float(g["event_timestamp_seconds"]))]
            trace_rows.append({
                "match_id": match, "video_filename": videos[match], "group_uid": group,
                "best_timestamp_seconds": row["best_timestamp_seconds"], "temporal_score_lomo": row["rank_score"],
                "temporal_rank_full_pool": rank_maps[match][group],
                "weak_train_label_unconfirmed": row["weak_label"],
                "delivery_start_seconds": row["delivery_start_seconds"], "delivery_end_seconds": row["delivery_end_seconds"],
                "selected_existing_top60": bool(selected), "temporal_rank_existing_top60": selected["temporal_rank"] if selected else "",
                "temporal_gap_rank_top80_extension": gap_rank[match].get(group, ""),
                "visual_score_lomo_if_scored": scores.get("visual_score_goal", ""),
                "visual_rank_within_common_top60": common_ranks["visual_score_goal"].get(group, ""),
                "fusion_score_lomo_if_scored": scores.get("fusion_score_goal", ""),
                "fusion_rank_within_common_top60": common_ranks["fusion_score_goal"].get(group, ""),
                "rrf_rank_within_common_top60_diagnostic_only": rrf_rank.get(group, ""),
                "goal_catalog_ids_covered": ";".join(associated_goals),
                "selection_stage_exclusion": exclusion,
                "gap_suppressor_group_uid": suppressor["source_interval_id"] if suppressor else "",
                "rrf_nms_full_pool_rank_available": False,
                "manual_candidate_label": "PENDING_HUMAN_REVIEW",
            })

    # Choose a representative game deterministically: most recoverable catalog
    # goals excluded from the operational top-60, then highest goal count, then ID.
    misses_by_match: dict[str, int] = defaultdict(int)
    for item in missed_goal_details:
        misses_by_match[item["goal"]["match_id"]] += 1
    representative = sorted(matches, key=lambda m: (-misses_by_match[m], -len(goals_by_match[m]), m))[0]
    queued: dict[tuple[str, str], dict[str, object]] = {}

    def queue_candidate(match: str, group: str, role: str, related_goal: str = "") -> None:
        key = (match, group)
        source = pool_group[key]
        rank = rank_maps[match][group]
        item = queued.setdefault(key, {
            "case_id": f"{match}__{group.split('::')[-1]}", "match_id": match,
            "video_filename": videos[match], "group_uid": group,
            "best_timestamp_seconds": source["best_timestamp_seconds"],
            "delivery_start_seconds": source["delivery_start_seconds"], "delivery_end_seconds": source["delivery_end_seconds"],
            "temporal_score_lomo": source["rank_score"], "temporal_rank_full_pool": rank,
            "top60_rank": next((r["temporal_rank"] for r in shortlist_sorted[match] if r["source_interval_id"] == group), ""),
            "weak_train_label_unconfirmed": source["weak_label"], "review_role": role,
            "related_goal_catalog_ids": related_goal, "manual_status": "PENDING",
            "catalog_goals_covered": "", "catalog_goal_timestamps_covered": "",
            "candidate_frames_csv": detector_sources.get(match, ""),
            "manual_goal_label": "", "ball_visible": "", "ball_proposal_quality": "",
            "goal_structure_visible": "", "candidate_goal_association": "",
            "offensive_context": "", "camera_motion": "", "occlusion_or_aerial": "",
            "selection_stage_observation": "", "cause_tags": "", "diagnosis_confidence": "",
            "reviewer_notes": "", "evidence_path": "",
        })
        covered = [g for g in goals_by_match[match] if covers(source, float(g["event_timestamp_seconds"]))]
        item["catalog_goals_covered"] = ";".join(g["goal_catalog_id"] for g in covered)
        item["catalog_goal_timestamps_covered"] = ";".join(g["event_timestamp_seconds"] for g in covered)
        if role not in item["review_role"].split(";"):
            item["review_role"] += ";" + role
        if related_goal and related_goal not in item["related_goal_catalog_ids"].split(";"):
            item["related_goal_catalog_ids"] = ";".join(filter(None, (item["related_goal_catalog_ids"], related_goal)))

    for row in shortlist_sorted[representative]:
        covered = [g for g in goals_by_match[representative] if covers(row, float(g["event_timestamp_seconds"]))]
        related = ";".join(g["goal_catalog_id"] for g in covered)
        queue_candidate(representative, row["source_interval_id"], "REPRESENTATIVE_MATCH_TOP60", related)
    for item in missed_goal_details:
        goal, best = item["goal"], item["best"]
        if best:
            queue_candidate(goal["match_id"], best["group_uid"], "RECOVERABLE_GOAL_BELOW_TOP60", goal["goal_catalog_id"])
            rank = rank_maps[goal["match_id"]][best["group_uid"]]
            comparisons = [r for r in shortlist_sorted[goal["match_id"]]
                           if rank_maps[goal["match_id"]][r["source_interval_id"]] < rank
                           and all(not covers(r, float(g["event_timestamp_seconds"])) for g in goals_by_match[goal["match_id"]])]
            for candidate in comparisons[:args.pair_comparators]:
                queue_candidate(goal["match_id"], candidate["source_interval_id"], "PAIRED_HIGHER_RANK_UNCONFIRMED", goal["goal_catalog_id"])
    for match, rows in shortlist_sorted.items():
        if match == representative:
            continue
        for position in (1, 15, 30):
            if len(rows) >= position:
                row = rows[position - 1]
                queue_candidate(match, row["source_interval_id"], "TOP30_CONTROL_UNCONFIRMED")

    annotation_rows = []
    for match, goals in sorted(goals_by_match.items()):
        annotation_rows.append({
            "annotation_level": "MATCH", "match_id": match, "group_uid": "", "video_filename": videos[match],
            "secondary_goal_status": "PENDING", "definition_version": "written_visual_review_v1",
            "evidence_path": "", "reviewer_notes": "",
        })
    for (match, group), item in sorted(queued.items()):
        annotation_rows.append({
            "annotation_level": "CANDIDATE", "match_id": match, "group_uid": group,
            "video_filename": videos[match], "secondary_goal_status": "PENDING",
            "definition_version": "written_visual_review_v1", "evidence_path": "",
            "reviewer_notes": "Avaliar estrutura secundÃ¡ria na janela especÃ­fica; presenÃ§a de duas caixas no mesmo frame nÃ£o basta.",
        })

    baseline_rows = []
    total_goals = len(goal_rows)
    for k in LIMITS:
        chosen = {m: gap_selected_by_match[m][:k] for m in shortlist_sorted}
        covered = sum(any(covers(r, float(g["goal_timestamp_seconds"])) for r in chosen[g["match_id"]]) for g in goal_rows)
        minutes, segments = 0.0, 0
        for match, rows in chosen.items():
            m, n = unique_minutes(rows)
            minutes += m
            segments += n
        baseline_rows.append({
            "method": "TEMPORAL_RF_LOMO_GAP45", "k": k,
            "catalog_goals": total_goals, "covered_goals": covered,
            "pool_recoverable_goals": sum(bool(r["pool_recoverable"]) for r in goal_rows),
            "recall_catalog": f"{covered / total_goals:.6f}",
            "recall_conditional_recoverable": f"{covered / max(1, sum(bool(r['pool_recoverable']) for r in goal_rows)):.6f}",
            "selected_candidates": sum(map(len, chosen.values())), "merged_delivery_segments": segments,
            "unique_review_minutes_total": f"{minutes:.3f}",
            "unique_review_minutes_per_match": f"{minutes / len(chosen):.3f}",
            "human_review_time_measured": False,
            "policy_status": "EXISTING_TOP60" if k <= 60 else "DESCRIPTIVE_EXTENSION_BEYOND_OPERATIONAL_TOP60",
        })

    args.output.mkdir(parents=True)
    write_csv(args.output / "case_manifest.csv", goal_rows)
    write_csv(args.output / "selection_trace.csv", trace_rows)
    write_csv(args.output / "manual_case_audit.csv", list(queued.values()))
    write_csv(args.output / "secondary_goal_annotations.csv", annotation_rows)
    pair_rows: list[dict[str, object]] = []
    for item in missed_goal_details:
        goal, best = item["goal"], item["best"]
        if not best:
            continue
        match, goal_uid = goal["match_id"], best["group_uid"]
        goal_features = feature_by_group.get((match, goal_uid), {})
        for candidate in shortlist_sorted[match]:
            candidate_uid = candidate["source_interval_id"]
            if rank_maps[match][candidate_uid] >= rank_maps[match][goal_uid]:
                continue
            if any(covers(candidate, float(g["event_timestamp_seconds"])) for g in goals_by_match[match]):
                continue
            candidate_features = feature_by_group.get((match, candidate_uid), {})
            pair = {
                "goal_catalog_id": goal["goal_catalog_id"], "match_id": match,
                "goal_group_uid": goal_uid, "goal_temporal_rank_full_pool": rank_maps[match][goal_uid],
                "goal_temporal_score_lomo": best["rank_score"], "candidate_group_uid": candidate_uid,
                "candidate_temporal_rank_full_pool": rank_maps[match][candidate_uid],
                "candidate_temporal_score_lomo": candidate_features.get("rank_score", ""),
                "candidate_weak_label_unconfirmed": candidate_features.get("weak_label", ""),
                "pair_label_status": "PENDING_HUMAN_CONFIRMATION_NO_GOAL_LABEL_INFERRED",
            }
            for field in feature_fields:
                if field in {"group_uid", "match_id", "best_timestamp_seconds", "weak_label", "rank_score"}:
                    continue
                left, right = goal_features.get(field, ""), candidate_features.get(field, "")
                pair[f"goal_feature_{field}"] = left
                pair[f"candidate_feature_{field}"] = right
                try:
                    pair[f"delta_candidate_minus_goal_{field}"] = f"{float(right) - float(left):.8g}"
                except (ValueError, TypeError):
                    pair[f"delta_candidate_minus_goal_{field}"] = ""
            pair_rows.append(pair)
            if sum(row["goal_catalog_id"] == goal["goal_catalog_id"] for row in pair_rows) >= args.pair_comparators:
                break
    if pair_rows:
        write_csv(args.output / "pairwise_rank_diagnostic.csv", pair_rows)
    write_csv(args.output / "coverage_baseline.csv", baseline_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; persisted LOMO predictions and detector outputs; no VALIDATION/TEST inputs; no new mining or training",
        "catalog_goals": total_goals, "full_pool_groups": len(predictions),
        "recoverable_goals": sum(bool(r["pool_recoverable"]) for r in goal_rows),
        "covered_temporal_top60": sum(bool(r["covered_by_existing_top60"]) for r in goal_rows),
        "representative_match_id": representative,
        "representative_match_top60_candidates": len(shortlist_sorted[representative]),
        "manual_case_rows": len(queued), "secondary_goal_annotations_pending": len(annotation_rows),
        "delivery_window": "20 seconds before to 40 seconds after, clipped to source video duration",
        "baseline_scores": "Existing temporal RF LOMO scores and frozen top-60 shortlist. Human review time is not inferred from video duration.",
        "known_limitations": [
            "weak_label=0 is not a verified NO_GOAL label",
            "full-pool visual/fusion/RRF ranks are not present; RRF diagnostics exist only within the common Top-60",
            "secondary_goal annotations and manual case labels are pending visual review",
            "detector rows are candidate-filtered historical proposals, not independent tracker identities",
        ],
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (args.rank_predictions, args.features, args.shortlist, args.score_audit, args.goal_catalog, args.manual_misses)},
        "input_row_counts": {"rank_predictions": len(predictions), "features": len(features), "shortlist": len(shortlist), "score_audit": len(score_audit), "catalog": len(catalog), "prior_manual_misses": len(previous_misses)},
    }
    (args.output / "coverage_baseline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "decision_summary.md").write_text(
        "# DiagnÃ³stico de erros de ranking â€” preparaÃ§Ã£o TRAIN\n\n"
        f"Partida escolhida deterministicamente para revisar a shortlist completa: **{representative}** ({len(shortlist_sorted[representative])} candidatos).\n\n"
        f"O pool persistido cobre {summary['recoverable_goals']}/{total_goals} gols; o Top-60 temporal cobre {summary['covered_temporal_top60']}/{total_goals}. "
        "As curvas e minutos de entrega estÃ£o em `coverage_baseline.csv`; minutos Ãºnicos descrevem duraÃ§Ã£o de vÃ­deo, nÃ£o tempo humano.\n\n"
        "A fila `manual_case_audit.csv` mantÃ©m todos os 60 candidatos da partida escolhida, os casos de gols recuperÃ¡veis fora do Top-60 e comparadores determinÃ­sticos. "
        "RÃ³tulos ainda estÃ£o pendentes: `weak_label=0` nÃ£o Ã© tratado como negativo confirmado. Secondary goal tambÃ©m permanece pendente de confirmaÃ§Ã£o visual.\n\n"
        "NÃ£o hÃ¡ scores visuais/fusÃ£o/RRF de pool completo em TRAIN entre as entradas verificadas; portanto a etapa nÃ£o fabrica ranks para essas fontes. "
        "O resultado Ã© preparaÃ§Ã£o reproduzÃ­vel, nÃ£o uma conclusÃ£o causal nem avaliaÃ§Ã£o final.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
