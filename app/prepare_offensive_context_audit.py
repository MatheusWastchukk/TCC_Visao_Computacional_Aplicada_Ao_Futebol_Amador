"""Prepara uma auditoria manual de contexto em falsos candidatos TRAIN."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path


FIELDS = [
    "audit_id", "event_id", "match_id", "video_filename", "delivery_start_seconds",
    "delivery_end_seconds", "temporal_rank", "rank_score", "context_label",
    "annotation_status", "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def covers_goal(candidate: dict[str, str], goals: list[float]) -> bool:
    start, end = float(candidate["delivery_start_seconds"]), float(candidate["delivery_end_seconds"])
    return any(start <= goal <= end for goal in goals)


def select_false_candidates(candidates: list[dict[str, str]], goals_by_match: dict[str, list[float]], limit: int) -> list[dict[str, str]]:
    """Seleciona ranks altos sem gol catalogado, alternando partidas para evitar viés."""
    by_match: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for row in sorted(candidates, key=lambda item: (item["match_id"], int(item["temporal_rank"]))):
        if not covers_goal(row, goals_by_match[row["match_id"]]):
            by_match[row["match_id"]].append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < limit and any(by_match.values()):
        for match_id in sorted(by_match):
            if by_match[match_id] and len(selected) < limit:
                selected.append(by_match[match_id].popleft())
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara falsos candidatos TRAIN para auditoria de contexto ofensivo.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if not 100 <= args.limit <= 150:
        parser.error("--limit deve permanecer entre 100 e 150 para a auditoria inicial.")
    candidates = read_csv(args.candidates)
    if not candidates or any(row.get("split") != "train" for row in candidates):
        raise AssertionError("A auditoria aceita somente candidatos TRAIN.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(args.goal_timestamps):
        if row.get("split") == "train" and row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds"):
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    selected = select_false_candidates(candidates, goals_by_match, args.limit)
    if len(selected) != args.limit:
        raise ValueError(f"Há somente {len(selected)} falsos candidatos elegíveis; eram necessários {args.limit}.")
    rows = [{
        "audit_id": f"offensive_context_{index:03d}", "event_id": row["event_id"],
        "match_id": row["match_id"], "video_filename": row["video_filename"],
        "delivery_start_seconds": row["delivery_start_seconds"], "delivery_end_seconds": row["delivery_end_seconds"],
        "temporal_rank": row["temporal_rank"], "rank_score": row["rank_score"],
        "context_label": "", "annotation_status": "pending", "notes": "",
    } for index, row in enumerate(selected, start=1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"itens={len(rows)}")
    print(f"partidas={len({row['match_id'] for row in rows})}")
    print(f"manifesto={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
