"""Audita perdas de shortlist e políticas de seleção em TRAIN, sem retreino."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para salvar: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    output: list[list[float]] = []
    for start, end in sorted(intervals):
        if not output or start > output[-1][1]:
            output.append([start, end])
        else:
            output[-1][1] = max(output[-1][1], end)
    return [(start, end) for start, end in output]


def greedy_gap(rows: list[dict[str, str]], gap: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        center = float(row["best_timestamp_seconds"])
        if all(abs(center - float(other["best_timestamp_seconds"])) >= gap for other in selected):
            selected.append(row)
    return selected


def greedy_nms(rows: list[dict[str, str]], threshold: float, before: float, after: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        center = float(row["best_timestamp_seconds"])
        start, end = center - before, center + after
        suppress = False
        for other in selected:
            other_center = float(other["best_timestamp_seconds"])
            other_start, other_end = other_center - before, other_center + after
            overlap = max(0.0, min(end, other_end) - max(start, other_start))
            union = (end - start) + (other_end - other_start) - overlap
            if union > 0 and overlap / union > threshold:
                suppress = True
                break
        if not suppress:
            selected.append(row)
    return selected


def temporal_clusters(rows: list[dict[str, str]], max_gap: float) -> list[dict[str, str]]:
    chronological = sorted(rows, key=lambda row: float(row["best_timestamp_seconds"]))
    clusters: list[list[dict[str, str]]] = []
    for row in chronological:
        if not clusters or float(row["best_timestamp_seconds"]) - float(clusters[-1][-1]["best_timestamp_seconds"]) > max_gap:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    return [max(cluster, key=lambda row: float(row["rank_score"])) for cluster in clusters]


def temporal_cluster_top_n(rows: list[dict[str, str]], max_gap: float, top_n: int) -> list[dict[str, str]]:
    chronological = sorted(rows, key=lambda row: float(row["best_timestamp_seconds"]))
    clusters: list[list[dict[str, str]]] = []
    for row in chronological:
        if not clusters or float(row["best_timestamp_seconds"]) - float(clusters[-1][-1]["best_timestamp_seconds"]) > max_gap:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    selected = [row for cluster in clusters for row in sorted(cluster, key=lambda item: float(item["rank_score"]), reverse=True)[:top_n]]
    return sorted(selected, key=lambda row: float(row["rank_score"]), reverse=True)


def greedy_mmr(rows: list[dict[str, str]], limit: int, relevance_weight: float) -> list[dict[str, str]]:
    if not rows:
        return []
    scores = [float(row["rank_score"]) for row in rows]
    low, high = min(scores), max(scores)
    scale = high - low or 1.0
    selected: list[dict[str, str]] = []
    remaining = list(rows)
    while remaining and len(selected) < limit:
        def mmr(row: dict[str, str]) -> float:
            relevance = (float(row["rank_score"]) - low) / scale
            if not selected:
                novelty = 1.0
            else:
                nearest = min(abs(float(row["best_timestamp_seconds"]) - float(other["best_timestamp_seconds"])) for other in selected)
                novelty = min(nearest / 180.0, 1.0)
            return relevance_weight * relevance + (1.0 - relevance_weight) * novelty
        chosen = max(remaining, key=mmr)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita ranking e diversidade temporal só em TRAIN.")
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--gaps", type=float, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 45])
    parser.add_argument("--window-before", type=float, default=20)
    parser.add_argument("--window-after", type=float, default=40)
    parser.add_argument("--limits", type=int, nargs="+", default=[60, 70, 80, 100, 999999])
    parser.add_argument("--selection-methods", action="store_true", help="Executa Temporal NMS, clustering, MMR, ensemble e RRF em TRAIN.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")

    predictions = [row for row in read_csv(args.rank_predictions) if row.get("model") == args.model]
    catalog = read_csv(args.goal_timestamps)
    if not predictions or not catalog:
        raise ValueError("Predições ou catálogo estão vazios.")
    if any(row.get("split", "").lower() != "train" for row in catalog):
        raise AssertionError("Este experimento aceita somente timestamps TRAIN.")
    goals: dict[str, list[float]] = defaultdict(list)
    filenames: dict[str, str] = {}
    for row in catalog:
        filenames[row["match_id"]] = row["video_filename"]
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    ranked: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        ranked[row["match_id"]].append(row)
    if set(goals) != set(ranked):
        raise ValueError("O ranking e o catálogo não contêm o mesmo conjunto de partidas.")
    for match in ranked:
        ranked[match].sort(key=lambda row: float(row["rank_score"]), reverse=True)

    all_predictions = read_csv(args.rank_predictions)
    rankings_by_model: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in all_predictions:
        rankings_by_model[row["model"]][row["match_id"]].append(row)
    for model_rankings in rankings_by_model.values():
        for match in model_rankings:
            model_rankings[match].sort(key=lambda row: float(row["rank_score"]), reverse=True)

    durations: dict[str, float] = {}
    for match, filename in filenames.items():
        video = cv2.VideoCapture(str(args.videos_dir / filename))
        if not video.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir vídeo TRAIN: {filename}")
        fps, frames = float(video.get(cv2.CAP_PROP_FPS)), float(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        if fps <= 0 or frames <= 0:
            raise ValueError(f"Metadados inválidos no vídeo: {filename}")
        durations[match] = frames / fps

    def clip(row: dict[str, str], match: str) -> tuple[float, float]:
        center = float(row["best_timestamp_seconds"])
        return max(0, center - args.window_before), min(durations[match], center + args.window_after)

    def covers(row: dict[str, str], timestamp: float) -> bool:
        start, end = clip(row, row["match_id"])
        return start <= timestamp <= end

    autopsy: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    total_goals = sum(map(len, goals.values()))
    for gap in args.gaps:
        selected_by_match: dict[str, list[dict[str, str]]] = {}
        for match, rows in ranked.items():
            chosen: list[dict[str, str]] = []
            for row in rows:
                center = float(row["best_timestamp_seconds"])
                if all(abs(center - float(prev["best_timestamp_seconds"])) >= gap for prev in chosen):
                    chosen.append(row)
            selected_by_match[match] = chosen
        for limit in args.limits:
            chosen = {match: rows[:limit] for match, rows in selected_by_match.items()}
            covered = sum(any(covers(row, timestamp) for row in chosen[match])
                          for match, times in goals.items() for timestamp in times)
            intervals = {match: merge([clip(row, match) for row in rows]) for match, rows in chosen.items()}
            unique_seconds = sum(end - start for values in intervals.values() for start, end in values)
            gap_rows.append({
                "gap_seconds": gap,
                "k_limit": "pool_diverso_completo" if limit >= max(map(len, selected_by_match.values())) else limit,
                "goals": total_goals,
                "covered_goals": covered,
                "recall": round(covered / total_goals, 6),
                "candidates": sum(map(len, chosen.values())),
                "merged_segments": sum(map(len, intervals.values())),
                "unique_review_minutes_per_match": round(unique_seconds / 60 / len(chosen), 3),
            })

    if args.selection_methods:
        method_rankings: dict[str, dict[str, list[dict[str, str]]]] = {}
        for model, model_rows in rankings_by_model.items():
            if set(model_rows) == set(ranked):
                method_rankings[f"modelo_{model}"] = model_rows
        if set(rankings_by_model) >= {"random_forest", "logistic_regression", "gradient_boosting"}:
            rrf_rankings: dict[str, list[dict[str, str]]] = {}
            models = ["random_forest", "logistic_regression", "gradient_boosting"]
            for match in ranked:
                rank_maps = [{row["group_uid"]: index for index, row in enumerate(rankings_by_model[model][match], 1)} for model in models]
                rows_by_uid = {row["group_uid"]: row for row in rankings_by_model[models[0]][match]}
                fused = []
                for uid, base_row in rows_by_uid.items():
                    score = sum(1.0 / (60 + ranks.get(uid, len(rows_by_uid) + 1)) for ranks in rank_maps)
                    fused.append({**base_row, "rank_score": str(score)})
                rrf_rankings[match] = sorted(fused, key=lambda row: float(row["rank_score"]), reverse=True)
            method_rankings["RRF_temporal_3_modelos"] = rrf_rankings

        selectors: dict[str, object] = {
            "gap_45s": lambda rows: greedy_gap(rows, 45.0),
            "Temporal_NMS_iou_0_25": lambda rows: greedy_nms(rows, 0.25, args.window_before, args.window_after),
            "Temporal_NMS_iou_0_50": lambda rows: greedy_nms(rows, 0.50, args.window_before, args.window_after),
            "Temporal_NMS_iou_0_75": lambda rows: greedy_nms(rows, 0.75, args.window_before, args.window_after),
            "MMR_lambda_0_50": lambda rows: greedy_mmr(rows, 60, 0.50),
            "MMR_lambda_0_70": lambda rows: greedy_mmr(rows, 60, 0.70),
            "MMR_lambda_0_85": lambda rows: greedy_mmr(rows, 60, 0.85),
        }
        for model_name, model_rows in method_rankings.items():
            for selector_name, selector in selectors.items():
                if selector_name != "gap_45s" and not model_name.startswith("modelo_random_forest") and not model_name.startswith("RRF"):
                    continue
                if selector_name == "gap_45s" and model_name.startswith("modelo_"):
                    pass
                selection_rows.extend(evaluate_selection(
                    f"{model_name}__{selector_name}",
                    {match: selector(rows) for match, rows in model_rows.items()},
                    goals, durations, args.window_before, args.window_after, args.limits,
                ))
        for cluster_gap in [60.0, 90.0, 120.0, 180.0]:
            for top_n in [1, 2, 3, 5]:
                selection_rows.extend(evaluate_selection(
                    f"RF__cluster_gap_{int(cluster_gap)}s_top{top_n}",
                    {match: temporal_cluster_top_n(rows, cluster_gap, top_n) for match, rows in ranked.items()},
                    goals, durations, args.window_before, args.window_after, args.limits,
                ))

    baseline_gap = 45.0
    selected45: dict[str, list[dict[str, str]]] = {}
    for match, rows in ranked.items():
        chosen = []
        for row in rows:
            if all(abs(float(row["best_timestamp_seconds"]) - float(prev["best_timestamp_seconds"])) >= baseline_gap for prev in chosen):
                chosen.append(row)
        selected45[match] = chosen

    for match, times in goals.items():
        rows = ranked[match]
        for index, timestamp in enumerate(times, 1):
            pool_hits = [row for row in rows if covers(row, timestamp)]
            diverse_hits = [row for row in selected45[match] if covers(row, timestamp)]
            first_pool = pool_hits[0] if pool_hits else None
            first_diverse = diverse_hits[0] if diverse_hits else None
            suppressor = None
            if first_pool and not first_diverse:
                center = float(first_pool["best_timestamp_seconds"])
                suppressor = next((row for row in selected45[match]
                                   if abs(center - float(row["best_timestamp_seconds"])) < baseline_gap), None)
            autopsy.append({
                "match_id": match,
                "goal_index_in_catalog_order": index,
                "goal_timestamp_seconds": timestamp,
                "pool_covered_20_40": bool(pool_hits),
                "best_raw_rank_covering_goal": rows.index(first_pool) + 1 if first_pool else "",
                "diverse_45s_covering_goal": bool(diverse_hits),
                "best_rank_after_45s_covering_goal": selected45[match].index(first_diverse) + 1 if first_diverse else "",
                "covered_in_top60_45s": bool(diverse_hits and selected45[match].index(first_diverse) < 60),
                "covered_in_top80_45s": bool(diverse_hits and selected45[match].index(first_diverse) < 80),
                "loss_cause": ("pool_sem_cobertura_20_40" if not pool_hits else
                               "abaixo_top60_apos_diversidade" if diverse_hits and selected45[match].index(first_diverse) >= 60 else
                               "supressao_gap45" if not diverse_hits else "coberto_top60"),
                "suppressing_candidate_uid": suppressor.get("group_uid", "") if suppressor else "",
                "suppressor_rank_score": suppressor.get("rank_score", "") if suppressor else "",
                "suppressor_timestamp_seconds": suppressor.get("best_timestamp_seconds", "") if suppressor else "",
            })

    args.output.mkdir(parents=True)
    write_csv(args.output / "goal_loss_autopsy_v1.csv", autopsy)
    write_csv(args.output / "gap_sweep_v1.csv", gap_rows)
    if selection_rows:
        write_csv(args.output / "selection_method_comparison_v1.csv", selection_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "somente TRAIN; sem retreinamento; TEST não foi acessado",
        "ranking_source": str(args.rank_predictions),
        "model": args.model,
        "matches": len(ranked),
        "catalog_goals": total_goals,
        "candidate_groups": sum(map(len, ranked.values())),
        "windows_for_review_seconds": {"before": args.window_before, "after": args.window_after},
        "candidate_pool_coverage": sum(bool(row["pool_covered_20_40"]) for row in autopsy),
        "diverse_45s_top60_coverage": sum(bool(row["covered_in_top60_45s"]) for row in autopsy),
        "loss_causes_top60_45s": dict(__import__("collections").Counter(row["loss_cause"] for row in autopsy if row["loss_cause"] != "coberto_top60")),
        "warning": "Resultados de desenvolvimento em TRAIN; sem inferência de generalização ou de confirmação oficial de gol.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def evaluate_selection(method: str, selected_by_match: dict[str, list[dict[str, str]]],
                       goals: dict[str, list[float]], durations: dict[str, float], before: float,
                       after: float, limits: list[int]) -> list[dict[str, object]]:
    total_goals = sum(map(len, goals.values()))
    rows: list[dict[str, object]] = []
    for limit in limits:
        chosen = {match: values[:limit] for match, values in selected_by_match.items()}
        covered = 0
        intervals: dict[str, list[tuple[float, float]]] = {}
        for match, candidates in chosen.items():
            match_intervals = []
            for candidate in candidates:
                center = float(candidate["best_timestamp_seconds"])
                start, end = max(0.0, center - before), min(durations[match], center + after)
                match_intervals.append((start, end))
            intervals[match] = merge(match_intervals)
        for match, times in goals.items():
            for timestamp in times:
                if any(max(0.0, float(row["best_timestamp_seconds"]) - before) <= timestamp <= min(durations[match], float(row["best_timestamp_seconds"]) + after) for row in chosen[match]):
                    covered += 1
        unique_seconds = sum(end - start for values in intervals.values() for start, end in values)
        rows.append({
            "method": method,
            "k_limit": "pool_diverso_completo" if limit >= max(map(len, selected_by_match.values())) else limit,
            "catalog_goals": total_goals,
            "covered_goals": covered,
            "recall": round(covered / total_goals, 6),
            "candidates": sum(map(len, chosen.values())),
            "merged_segments": sum(map(len, intervals.values())),
            "unique_review_minutes_per_match": round(unique_seconds / 60 / len(chosen), 3),
        })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
