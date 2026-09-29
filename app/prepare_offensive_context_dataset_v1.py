"""Monta o dataset TRAIN do Context Gate a partir de pool, gols e auditoria."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path


AUDIT_TO_DECISION = {
    "SEM_ACAO_OFENSIVA": "REJECT_NO_ACTION",
    "OUTRO": "REJECT_NO_ACTION",
    "ACAO_OFENSIVA_MAS_NAO_GOL": "KEEP_FOR_REVIEW",
    "GOL_ANULADO_POR_IMPEDIMENTO": "KEEP_FOR_REVIEW",
    "INCERTO": "KEEP_FOR_REVIEW",
    "GOAL_CATALOGO_AUSENTE": "KEEP_FOR_REVIEW",
}
FIELDS = [
    "context_id", "event_id", "match_id", "video_filename", "delivery_start_seconds",
    "delivery_end_seconds", "temporal_rank", "rank_score", "goal_linked",
    "context_label", "context_decision", "label_source", "annotation_status", "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def covers_goal(candidate: dict[str, str], goals: list[float]) -> bool:
    return any(float(candidate["delivery_start_seconds"]) <= goal <= float(candidate["delivery_end_seconds"]) for goal in goals)


def round_robin(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    by_match: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: (item["match_id"], int(item["temporal_rank"]))):
        by_match[row["match_id"]].append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < limit and any(by_match.values()):
        for match_id in sorted(by_match):
            if by_match[match_id] and len(selected) < limit:
                selected.append(by_match[match_id].popleft())
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara dataset TRAIN de contexto ofensivo.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--goal-timestamps", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=360)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    if not 300 <= args.target_size <= 500:
        parser.error("--target-size deve permanecer entre 300 e 500.")
    candidates = read_csv(args.candidates)
    if not candidates or any(row.get("split") != "train" for row in candidates):
        raise AssertionError("O dataset de contexto aceita somente candidatos TRAIN.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for timestamps_path in args.goal_timestamps:
        for row in read_csv(timestamps_path):
            if row.get("split") == "train" and row.get("annotation_status") == "completed" and row.get("event_timestamp_seconds"):
                goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    audit_by_event = {row["event_id"]: row for row in read_csv(args.audit_annotations) if row.get("annotation_status") == "completed"}
    selected: dict[str, tuple[dict[str, str], str, str, str]] = {}
    for candidate in candidates:
        audit = audit_by_event.get(candidate["event_id"])
        if audit and audit.get("context_label") in AUDIT_TO_DECISION:
            selected[candidate["event_id"]] = (candidate, audit["context_label"], AUDIT_TO_DECISION[audit["context_label"]], "audit_manual")
        elif covers_goal(candidate, goals_by_match[candidate["match_id"]]):
            selected[candidate["event_id"]] = (candidate, "GOAL_CATALOGADO", "KEEP_FOR_REVIEW", "catalogo")
    remaining = [
        row for row in candidates
        if row["event_id"] not in selected and 20 <= int(row["temporal_rank"]) <= 45
    ]
    for candidate in round_robin(remaining, max(0, args.target_size - len(selected))):
        selected[candidate["event_id"]] = (candidate, "", "", "amostra_intermediaria")
    if len(selected) < args.target_size:
        raise ValueError(f"Foram selecionados apenas {len(selected)} itens; alvo: {args.target_size}.")
    rows = []
    for index, (candidate, label, decision, source) in enumerate(sorted(selected.values(), key=lambda item: (item[0]["match_id"], int(item[0]["temporal_rank"]))), start=1):
        rows.append({
            "context_id": f"offensive_context_dataset_{index:03d}", "event_id": candidate["event_id"],
            "match_id": candidate["match_id"], "video_filename": candidate["video_filename"],
            "delivery_start_seconds": candidate["delivery_start_seconds"], "delivery_end_seconds": candidate["delivery_end_seconds"],
            "temporal_rank": candidate["temporal_rank"], "rank_score": candidate["rank_score"],
            "goal_linked": str(covers_goal(candidate, goals_by_match[candidate["match_id"]])),
            "context_label": label, "context_decision": decision, "label_source": source,
            "annotation_status": "completed" if decision else "pending", "notes": "",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["label_source"]] += 1
    print(f"itens={len(rows)}")
    print(f"origens={dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
