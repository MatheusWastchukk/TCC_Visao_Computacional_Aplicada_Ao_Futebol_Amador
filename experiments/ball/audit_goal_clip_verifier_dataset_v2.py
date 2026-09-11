"""Audita integridade e prepara revisão amostral dos negativos temporais do V2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita o dataset TRAIN de clipes de gol V2.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-exclusion-seconds", type=float, default=90.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    rows = read_csv(args.dataset)
    catalog = read_csv(args.catalog)
    if any(row["split"] != "train" for row in rows):
        raise AssertionError("A auditoria aceita somente TRAIN; VALIDATION e TEST permanecem fechados.")
    goals_by_match: dict[str, list[float]] = defaultdict(list)
    for row in catalog:
        if row["annotation_status"] == "completed":
            goals_by_match[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    for pair_id, pair_rows in pairs.items():
        if len(pair_rows) != 2 or {row["label"] for row in pair_rows} != {"GOAL", "NO_GOAL"}:
            raise ValueError(f"Par inválido: {pair_id}")
        if len({row["match_id"] for row in pair_rows}) != 1:
            raise ValueError(f"Par em partidas diferentes: {pair_id}")
    negatives = [row for row in rows if row["label"] == "NO_GOAL"]
    for row in negatives:
        center = float(row["center_seconds"])
        distance = min(abs(center - timestamp) for timestamp in goals_by_match[row["match_id"]])
        if distance < args.goal_exclusion_seconds:
            raise ValueError(f"Negativo dentro da exclusão de gol: {row['clip_id']} ({distance:.3f}s)")
    safe_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in negatives:
        if row["source_type"] == "safe_temporal_negative":
            safe_by_match[row["match_id"]].append(row)
    queue: list[dict[str, str]] = []
    for match_id, candidates in sorted(safe_by_match.items()):
        selected = min(candidates, key=lambda row: hashlib.sha256(row["clip_id"].encode("utf-8")).hexdigest())
        queue.append({
            "event_id": selected["clip_id"], "match_id": match_id, "video_filename": selected["video_filename"],
            "split": "train", "event_timestamp_seconds": selected["center_seconds"],
            "expected_label": "NO_GOAL", "source_type": selected["source_type"],
            "pair_id": selected["pair_id"], "annotation_status": "pending",
            "notes": "Auditoria humana cega de negativo temporal; marcar G se houver gol, N se não houver, U se não for avaliável.",
        })
    args.output.mkdir(parents=True)
    fields = list(queue[0])
    with (args.output / "negative_audit_queue_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(queue)
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN", "clips": len(rows),
        "pairs": len(pairs), "negative_sources": dict(Counter(row["source_type"] for row in negatives)),
        "safe_negative_audit_queue": len(queue), "queue_sampling": "um negativo temporal seguro por partida",
        "goal_exclusion_seconds_verified": args.goal_exclusion_seconds,
        "warning": "A fila exige revisão humana antes de usar negativos temporais para treinamento ou métricas.",
    }
    (args.output / "integrity_summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
