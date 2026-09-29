"""Compara selecao, RRF, NMS e cortes adaptativos no pool visual completo TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SCORE_FIELDS = ("rank_score", "visual_score_goal", "fusion_score_goal")
LIMITS = (20, 30, 40, 50, 60, 80, 100)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gap_select(rows: list[dict[str, str]], seconds: float = 45.0) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    for row in rows:
        center = float(row["delivery_start_seconds"]) + 20.0
        if all(abs(center - (float(other["delivery_start_seconds"]) + 20.0)) >= seconds for other in chosen):
            chosen.append(row)
    return chosen


def iou_nms(rows: list[dict[str, str]], threshold: float) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    for row in rows:
        start, end = float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])
        suppressed = False
        for other in chosen:
            other_start, other_end = float(other["delivery_start_seconds"]), float(other["delivery_end_seconds"])
            intersection = max(0.0, min(end, other_end) - max(start, other_start))
            union = (end - start) + (other_end - other_start) - intersection
            if union > 0 and intersection / union > threshold:
                suppressed = True
                break
        if not suppressed:
            chosen.append(row)
    return chosen


def measure(method: str, chosen: dict[str, list[dict[str, str]]], goals: dict[str, list[float]], limit: int | None) -> dict[str, object]:
    selected = {match: rows if limit is None else rows[:limit] for match, rows in chosen.items()}
    covered = sum(
        any(float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"]) for row in selected[match])
        for match, times in goals.items() for goal in times
    )
    segments = 0
    seconds = 0.0
    for rows in selected.values():
        merged: list[list[float]] = []
        for start, end in sorted((float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])) for row in rows):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        segments += len(merged)
        seconds += sum(end - start for start, end in merged)
    total = sum(map(len, goals.values()))
    return {
        "method": method,
        "k": "adaptive" if limit is None else limit,
        "catalog_goals": total,
        "covered_goals": covered,
        "recall": f"{covered / total:.6f}",
        "candidates": sum(map(len, selected.values())),
        "merged_segments": segments,
        "unique_review_minutes_per_match": f"{seconds / 60 / len(selected):.3f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Selecao de ranking no pool completo TRAIN, sem acessar VALIDATION ou TEST.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--utility-ltr-predictions", type=Path)
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--gap-seconds", type=float, nargs="+", default=[30, 45, 60])
    parser.add_argument("--fusion-thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    candidates = read_csv(args.candidate_scores)
    catalog = read_csv(args.goal_timestamps)
    score_fields = list(SCORE_FIELDS)
    # Permite comparar scores multi-subclip sobre o mesmo pool e as mesmas
    # janelas de entrega quando o CSV de entrada os contém.
    score_fields.extend(
        field for field in ("subclip_max_score", "fusion_subclip_max_score")
        if field in candidates[0] and field not in score_fields
    )
    if args.utility_ltr_predictions:
        ltr_rows = read_csv(args.utility_ltr_predictions)
        required_ltr = {"group_uid", "match_id", "rank_score"}
        if not ltr_rows or required_ltr - set(ltr_rows[0]):
            raise ValueError("Predicoes Clip Utility LTR sem group_uid, match_id ou rank_score.")
        ltr_scores = {(row["match_id"], row["group_uid"]): row["rank_score"] for row in ltr_rows}
        candidate_keys = {(row["match_id"], row["group_uid"]) for row in candidates}
        if len(ltr_scores) != len(ltr_rows) or candidate_keys != set(ltr_scores):
            raise ValueError("Scores Clip Utility LTR e reranker nao cobrem os mesmos grupos.")
        for row in candidates:
            row["ltr_utility_score"] = ltr_scores[(row["match_id"], row["group_uid"])]
        score_fields.append("ltr_utility_score")
    if not candidates or any(row.get("split", "train").lower() != "train" for row in catalog):
        raise AssertionError("Entrada vazia ou catálogo fora de TRAIN.")
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        by_match[row["match_id"]].append(row)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if set(by_match) != set(goals) or sum(map(len, by_match.values())) != 5742:
        raise ValueError("O pool completo precisa corresponder ao catÃ¡logo das 25 partidas TRAIN e conter 5.742 grupos.")

    rankings: dict[str, dict[str, list[dict[str, str]]]] = {field: {} for field in score_fields}
    rankings["rrf"] = {}
    rankings["rrf_3source"] = {}
    for match, rows in by_match.items():
        for field in score_fields:
            rankings[field][match] = sorted(rows, key=lambda row: float(row[field]), reverse=True)
        reciprocal: dict[str, float] = defaultdict(float)
        for field in score_fields:
            for rank, row in enumerate(rankings[field][match], start=1):
                reciprocal[row["group_uid"]] += 1.0 / (60 + rank)
        rankings["rrf"][match] = sorted(rows, key=lambda row: reciprocal[row["group_uid"]], reverse=True)
        reciprocal_3source: dict[str, float] = defaultdict(float)
        for field in score_fields[:3]:
            for rank, row in enumerate(rankings[field][match], start=1):
                reciprocal_3source[row["group_uid"]] += 1.0 / (60 + rank)
        rankings["rrf_3source"][match] = sorted(rows, key=lambda row: reciprocal_3source[row["group_uid"]], reverse=True)

    methods: dict[str, dict[str, list[dict[str, str]]]] = {}
    for name, by_game in rankings.items():
        for gap in args.gap_seconds:
            methods[f"{name}_gap{gap:g}s"] = {match: gap_select(rows, gap) for match, rows in by_game.items()}
    nms_rankings = ["rrf", "fusion_score_goal"] + (["ltr_utility_score"] if "ltr_utility_score" in rankings else [])
    for name in nms_rankings:
        for threshold in args.nms_iou:
            methods[f"{name}_nms_iou{threshold:.2f}"] = {
                match: iou_nms(rows, threshold) for match, rows in rankings[name].items()
            }

    for temporal, visual, fusion in ((20, 20, 20), (30, 20, 20), (40, 30, 30)):
        union_by_match: dict[str, list[dict[str, str]]] = {}
        for match in by_match:
            seen: set[str] = set()
            union_rows: list[dict[str, str]] = []
            for field, count in (("rank_score", temporal), ("visual_score_goal", visual), ("fusion_score_goal", fusion)):
                for row in rankings[field][match][:count]:
                    if row["group_uid"] not in seen:
                        seen.add(row["group_uid"])
                        union_rows.append(row)
            union_by_match[match] = gap_select(union_rows, 45.0)
        methods[f"union_{temporal}_{visual}_{fusion}_gap45s"] = union_by_match

    adaptive: dict[str, dict[str, list[dict[str, str]]]] = {}
    fusion_gap = methods["fusion_score_goal_gap45s"]
    for threshold in args.fusion_thresholds:
        adaptive[f"fusion_score_gte{threshold:.2f}_gap45s"] = {
            match: [row for row in rows if float(row["fusion_score_goal"]) >= threshold][:60]
            for match, rows in fusion_gap.items()
        }
    for ratio in (0.50, 0.70, 0.85, 0.95):
        selected: dict[str, list[dict[str, str]]] = {}
        for match, rows in fusion_gap.items():
            cutoff = float(rows[0]["fusion_score_goal"]) * ratio if rows else float("inf")
            selected[match] = [row for row in rows if float(row["fusion_score_goal"]) >= cutoff][:60]
        adaptive[f"fusion_relative_topscore_{ratio:.2f}_gap45s"] = selected
    for drop in (0.05, 0.10, 0.20):
        selected = {}
        for match, rows in fusion_gap.items():
            kept: list[dict[str, str]] = []
            for row in rows:
                if kept and float(kept[-1]["fusion_score_goal"]) - float(row["fusion_score_goal"]) >= drop:
                    break
                kept.append(row)
            selected[match] = kept[:60]
        adaptive[f"fusion_stop_score_drop_{drop:.2f}_gap45s"] = selected

    output_rows: list[dict[str, object]] = []
    for name, selection in methods.items():
        for limit in LIMITS:
            output_rows.append(measure(name, selection, goals, limit))
    for name, selection in adaptive.items():
        output_rows.append(measure(name, selection, goals, None))
    args.output.mkdir(parents=True)
    write_csv(args.output / "full_pool_selection_comparison_v1.csv", output_rows)
    summary = {
        "experiment_id": args.output.name,
        "scope": "TRAIN only; 5,742 complete-pool candidate scores; OOF visual/fusion scores; TEST not accessed",
        "catalog_goals": sum(map(len, goals.values())),
        "candidate_rows": len(candidates),
        "score_fields": score_fields,
        "utility_ltr_predictions": str(args.utility_ltr_predictions) if args.utility_ltr_predictions else None,
        "review_window": "delivery_start_seconds/delivery_end_seconds (nominally 20+40 seconds)",
        "nms_iou_thresholds": args.nms_iou,
        "gap_seconds": args.gap_seconds,
        "fusion_score_thresholds": args.fusion_thresholds,
        "adaptive_policy": "exploratory score thresholds/relative score drop on fusion ranking, after 45-second gap, cap 60",
        "warning": "Exploratory TRAIN selection comparison; no policy promoted or calibrated for VALIDATION/TEST.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
