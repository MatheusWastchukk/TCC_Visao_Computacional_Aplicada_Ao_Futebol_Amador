"""Mede, sem nova inferência, a cobertura temporal ao variar candidatos por partida."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hit:
    match_id: str
    timestamp: float
    score: float


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def require_fields(fields: list[str], required: set[str], label: str) -> None:
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{label} sem colunas obrigatórias: {sorted(missing)}")


def group_hits(hits: list[Hit], merge_gap_seconds: float, minimum_hits: int) -> list[list[Hit]]:
    groups: list[list[Hit]] = []
    for hit in sorted(hits, key=lambda item: item.timestamp):
        if not groups or hit.timestamp - groups[-1][-1].timestamp > merge_gap_seconds:
            groups.append([hit])
        else:
            groups[-1].append(hit)
    return [group for group in groups if len(group) >= minimum_hits]


def select_groups(groups: list[list[Hit]], limit: int, separation_seconds: float) -> list[list[Hit]]:
    selected: list[list[Hit]] = []
    for group in sorted(groups, key=lambda item: max(hit.score for hit in item), reverse=True):
        best_timestamp = max(group, key=lambda item: item.score).timestamp
        selected_timestamps = [max(other, key=lambda item: item.score).timestamp for other in selected]
        if any(abs(best_timestamp - other) < separation_seconds for other in selected_timestamps):
            continue
        selected.append(group)
        if len(selected) >= limit:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def union_duration(windows: list[tuple[float, float]]) -> float:
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for start, end in sorted(windows):
        if current_start is None or current_end is None:
            current_start, current_end = start, end
        elif start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Varia a capacidade do minerador usando hits TRAIN já preservados."
    )
    parser.add_argument("--hit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument("--merge-gap-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-hits", type=int, default=2)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=10.0)
    parser.add_argument("--clip-post-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if not args.candidate_limits or min(args.candidate_limits) < 1:
        raise ValueError("candidate-limits deve conter inteiros positivos.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}.")

    hits_by_match: dict[str, list[Hit]] = defaultdict(list)
    for path in args.hit_files:
        rows, fields = read_csv(path)
        require_fields(fields, {"match_id", "timestamp_seconds", "candidate_score"}, f"Hits em {path}")
        for row in rows:
            hits_by_match[row["match_id"]].append(
                Hit(row["match_id"], float(row["timestamp_seconds"]), float(row["candidate_score"]))
            )
    goals, goal_fields = read_csv(args.goal_timestamps)
    require_fields(
        goal_fields,
        {"goal_catalog_id", "match_id", "split", "event_timestamp_seconds", "annotation_status"},
        "Timestamps",
    )
    if any(row["split"] != "train" for row in goals):
        raise AssertionError("A varredura aceita somente TRAIN; TEST permanece fechado.")
    evaluable_goals = [
        row for row in goals if row["annotation_status"] == "completed" and row["event_timestamp_seconds"]
    ]
    if not evaluable_goals:
        raise ValueError("Não há gols avaliáveis no manifesto.")

    grouped = {
        match_id: group_hits(hits, args.merge_gap_seconds, args.minimum_hits)
        for match_id, hits in hits_by_match.items()
    }
    sweep_rows: list[dict[str, str]] = []
    match_rows: list[dict[str, str]] = []
    for limit in sorted(set(args.candidate_limits)):
        windows_by_match: dict[str, list[tuple[float, float]]] = {}
        selected_count = 0
        for match_id, groups in grouped.items():
            selected = select_groups(groups, limit, args.min_separation_seconds)
            windows_by_match[match_id] = [
                (
                    max(0.0, max(group, key=lambda item: item.score).timestamp - args.clip_pre_seconds),
                    max(group, key=lambda item: item.score).timestamp + args.clip_post_seconds,
                )
                for group in selected
            ]
            selected_count += len(selected)
        unique_review_seconds = sum(union_duration(windows) for windows in windows_by_match.values())
        covered_by_match: Counter[str] = Counter()
        total_by_match: Counter[str] = Counter(row["match_id"] for row in evaluable_goals)
        for goal in evaluable_goals:
            timestamp = float(goal["event_timestamp_seconds"])
            if any(start <= timestamp <= end for start, end in windows_by_match.get(goal["match_id"], [])):
                covered_by_match[goal["match_id"]] += 1
        covered = sum(covered_by_match.values())
        sweep_rows.append(
            {
                "max_candidates_per_match": str(limit),
                "selected_candidate_windows": str(selected_count),
                "unique_review_seconds": f"{unique_review_seconds:.3f}",
                "unique_review_hours": f"{unique_review_seconds / 3600.0:.3f}",
                "evaluable_catalog_goals": str(len(evaluable_goals)),
                "covered_catalog_goals": str(covered),
                "candidate_window_recall": f"{covered / len(evaluable_goals):.6f}",
            }
        )
        for match_id in sorted(total_by_match):
            total = total_by_match[match_id]
            match_rows.append(
                {
                    "max_candidates_per_match": str(limit),
                    "match_id": match_id,
                    "evaluable_catalog_goals": str(total),
                    "covered_catalog_goals": str(covered_by_match[match_id]),
                    "candidate_window_recall": f"{covered_by_match[match_id] / total:.6f}",
                }
            )

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "capacity_sweep_v1.csv", sweep_rows)
    write_csv(args.output / "capacity_sweep_per_match_v1.csv", match_rows)
    summary = {
        "experiment_id": "GOAL-EVENT-CANDIDATE-CAPACITY-SWEEP-V1",
        "scope": "somente TRAIN; reuso de hits já minerados",
        "hit_files": [str(path) for path in args.hit_files],
        "evaluable_catalog_goals": len(evaluable_goals),
        "parameters": {
            "candidate_limits": sorted(set(args.candidate_limits)),
            "merge_gap_seconds": args.merge_gap_seconds,
            "minimum_hits": args.minimum_hits,
            "min_separation_seconds": args.min_separation_seconds,
            "clip_pre_seconds": args.clip_pre_seconds,
            "clip_post_seconds": args.clip_post_seconds,
        },
        "note": (
            "Varredura de desenvolvimento sem nova inferência e sem anotação humana dos novos candidatos. "
            "Mede somente cobertura temporal em TRAIN; não é métrica final nem confirmação online de gol."
        ),
    }
    (args.output / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(sweep_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
