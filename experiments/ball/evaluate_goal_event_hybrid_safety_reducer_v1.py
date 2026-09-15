"""Avalia merge temporal, híbridos de segurança e perdas antes do top 60."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_duration(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Duração inválida no manifesto: {value}")


def candidate_id(row: dict[str, str]) -> str:
    return row.get("event_id") or row.get("candidate_id") or row["source_interval_id"]


def merge_intervals(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    intervals = sorted(
        (float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"]))
        for row in rows
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def covers(row: dict[str, str], goal: float) -> bool:
    return float(row["delivery_start_seconds"]) <= goal <= float(row["delivery_end_seconds"])


def select_hybrid(rows: list[dict[str, str]], fusion_core: int, temporal_extra: int) -> list[dict[str, str]]:
    selected = sorted(rows, key=lambda row: float(row["fusion_score_goal"]), reverse=True)[:fusion_core]
    selected_ids = {candidate_id(row) for row in selected}
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        if candidate_id(row) not in selected_ids:
            selected.append(row)
            selected_ids.add(candidate_id(row))
            if len(selected) == fusion_core + temporal_extra:
                break
    return selected


def select_diverse(rows: list[dict[str, str]], separation: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
        timestamp = float(row["best_timestamp_seconds"])
        if any(abs(timestamp - float(other["best_timestamp_seconds"])) < separation for other in selected):
            continue
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia redução híbrida com merge temporal, sem treinar pesos.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["train", "validation"], default="train")
    parser.add_argument("--fusion-core", type=int, default=30)
    parser.add_argument("--temporal-extras", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--skip-hybrids", action="store_true", help="Mede somente referências e auditoria, sem avaliar híbridos.")
    parser.add_argument("--ranked-groups", type=Path)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.fusion_core <= 0 or any(extra <= 0 for extra in args.temporal_extras):
        raise ValueError("O núcleo de fusão e os complementos temporais devem ser positivos.")

    candidates = read_csv(args.candidate_scores)
    catalog = read_csv(args.goal_timestamps)
    manifest = read_csv(args.split_manifest)
    if not candidates:
        raise ValueError("Não há candidatos para avaliar.")
    if any(row["split"] != args.expected_split for row in catalog):
        raise AssertionError("O catálogo contém split incompatível.")
    durations = {
        row["match_id"]: parse_duration(row["duration_catalog"])
        for row in manifest
        if row["split"] == args.expected_split
    }
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        if row["match_id"] not in durations:
            raise AssertionError("Os candidatos contêm partida fora do split esperado.")
        by_match[row["match_id"]].append(row)
    for row in catalog:
        if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]:
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    if set(by_match) != set(goals_by_match):
        raise AssertionError("Candidatos e catálogo devem conter as mesmas partidas.")

    methods: list[tuple[str, int | None, int | None]] = [("rank_score", None, None), ("fusion_top_30", 30, None)]
    if not args.skip_hybrids:
        methods.extend((f"hybrid_f{args.fusion_core}_t{extra}", args.fusion_core, extra) for extra in sorted(set(args.temporal_extras)))
    curve_rows: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    all_goals = sum(len(goals) for goals in goals_by_match.values())
    for name, core, extra in methods:
        selections: dict[str, list[dict[str, str]]] = {}
        for match_id, rows in by_match.items():
            if name == "rank_score":
                selections[match_id] = sorted(rows, key=lambda row: float(row["rank_score"]), reverse=True)[:60]
            elif name == "fusion_top_30":
                selections[match_id] = sorted(rows, key=lambda row: float(row["fusion_score_goal"]), reverse=True)[:30]
            else:
                selections[match_id] = select_hybrid(rows, int(core), int(extra))
        covered = 0
        total_unique_seconds = 0.0
        total_segments = 0
        total_selected = 0
        for match_id, selected in sorted(selections.items()):
            intervals = merge_intervals(selected)
            unique_seconds = sum(end - start for start, end in intervals)
            match_covered = sum(any(covers(row, goal) for row in selected) for goal in goals_by_match[match_id])
            covered += match_covered
            total_unique_seconds += unique_seconds
            total_segments += len(intervals)
            total_selected += len(selected)
            per_match_rows.append({
                "method": name,
                "match_id": match_id,
                "catalog_goals": len(goals_by_match[match_id]),
                "covered_goals": match_covered,
                "selected_candidates": len(selected),
                "segments_after_merge": len(intervals),
                "review_seconds_brutos": len(selected) * 60.0,
                "review_seconds_unique": f"{unique_seconds:.3f}",
                "review_minutes_unique": f"{unique_seconds / 60.0:.3f}",
                "match_duration_seconds": f"{durations[match_id]:.3f}",
                "reduction_percent_vs_match": f"{100.0 * (1.0 - unique_seconds / durations[match_id]):.3f}",
            })
        total_duration = sum(durations.values())
        curve_rows.append({
            "method": name,
            "catalog_goals": all_goals,
            "covered_goals": covered,
            "recall": f"{covered / all_goals:.6f}",
            "mean_candidates_per_match": f"{total_selected / len(by_match):.3f}",
            "mean_segments_per_match": f"{total_segments / len(by_match):.3f}",
            "mean_review_minutes_brutos": f"{total_selected / len(by_match):.3f}",
            "mean_review_minutes_unique": f"{total_unique_seconds / len(by_match) / 60.0:.3f}",
            "total_review_minutes_unique": f"{total_unique_seconds / 60.0:.3f}",
            "reduction_percent_vs_total_match_duration": f"{100.0 * (1.0 - total_unique_seconds / total_duration):.3f}",
        })

    audit_rows: list[dict[str, object]] = []
    if args.ranked_groups:
        ranked_groups = read_csv(args.ranked_groups)
        raw_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in ranked_groups:
            if row["match_id"] in durations:
                raw_by_match[row["match_id"]].append(row)
        for row in catalog:
            if row["annotation_status"] != "completed" or not row["event_timestamp_seconds"]:
                continue
            match_id, goal = row["match_id"], float(row["event_timestamp_seconds"])
            ranked = sorted(raw_by_match[match_id], key=lambda item: float(item["rank_score"]), reverse=True)
            covering_raw = [item for item in ranked if float(item["best_timestamp_seconds"]) - 20.0 <= goal <= float(item["best_timestamp_seconds"]) + 40.0]
            diverse = select_diverse(ranked, args.min_separation_seconds)
            covering_diverse = [item for item in diverse if float(item["best_timestamp_seconds"]) - 20.0 <= goal <= float(item["best_timestamp_seconds"]) + 40.0]
            if covering_raw and (not covering_diverse or diverse.index(covering_diverse[0]) + 1 > 60):
                first_raw = covering_raw[0]
                first_diverse = covering_diverse[0] if covering_diverse else None
                classification = "suppressed_by_diversity" if first_diverse is None else "below_top60_after_diversity"
                blockers = [item for item in diverse if abs(float(item["best_timestamp_seconds"]) - float(first_raw["best_timestamp_seconds"])) < args.min_separation_seconds]
                audit_rows.append({
                    "goal_catalog_id": row["goal_catalog_id"],
                    "match_id": match_id,
                    "goal_timestamp_seconds": f"{goal:.3f}",
                    "classification": classification,
                    "raw_covering_groups": len(covering_raw),
                    "best_raw_covering_rank": ranked.index(first_raw) + 1,
                    "best_raw_covering_group_uid": first_raw["group_uid"],
                    "best_raw_covering_timestamp_seconds": first_raw["best_timestamp_seconds"],
                    "best_raw_covering_score": first_raw["rank_score"],
                    "best_diverse_covering_rank": diverse.index(first_diverse) + 1 if first_diverse else "",
                    "diversity_blocker_count": len(blockers),
                    "nearest_diversity_blocker_rank": diverse.index(blockers[0]) + 1 if blockers else "",
                    "nearest_diversity_blocker_timestamp_seconds": blockers[0]["best_timestamp_seconds"] if blockers else "",
                    "nearest_diversity_blocker_score": blockers[0]["rank_score"] if blockers else "",
                })

    args.output.mkdir(parents=True)
    write_csv(args.output / "hybrid_merge_curve_v1.csv", curve_rows, list(curve_rows[0]))
    write_csv(args.output / "per_match_hybrid_merge_v1.csv", per_match_rows, list(per_match_rows[0]))
    if args.ranked_groups:
        fields = ["goal_catalog_id", "match_id", "goal_timestamp_seconds", "classification", "raw_covering_groups", "best_raw_covering_rank", "best_raw_covering_group_uid", "best_raw_covering_timestamp_seconds", "best_raw_covering_score", "best_diverse_covering_rank", "diversity_blocker_count", "nearest_diversity_blocker_rank", "nearest_diversity_blocker_timestamp_seconds", "nearest_diversity_blocker_score"]
        write_csv(args.output / "top60_gap_audit_v1.csv", audit_rows, fields)
    summary = {
        "experiment_id": args.output.name,
        "scope": f"somente {args.expected_split.upper()}; escores congelados, sem retreinamento",
        "delivery_window": "20+40",
        "methods": [row["method"] for row in curve_rows],
        "selection_rule_for_validation": "Entre F30+T5/T10/T15/T20, escolher em TRAIN a maior cobertura com redução de ao menos 20% dos minutos únicos frente ao top 60 temporal; desempatar pelo menor tempo único.",
        "top60_gap_audit_cases": len(audit_rows),
        "warning": "TEST não foi acessado. VALIDATION não deve ser usada para escolher a configuração híbrida.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "curve": curve_rows, "audit": audit_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
