"""Prepara o lote inicial do GOAL-EVENT-DEVSET-V1 usando somente TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = [
    "event_id", "source_file", "source_interval_id", "source_interval_label",
    "match_id", "video_filename", "split", "event_timestamp_seconds",
    "event_time_hhmmss", "ground_truth", "ball_visible", "goal_visible",
    "spatial_candidate", "relative_motion_valid", "identity_stable",
    "center_visible_after", "ball_at_center_after", "players_reset",
    "kickoff_observed", "restart_time", "scoreboard_visible", "score_changed",
    "not_evaluable_reason", "annotation_status", "notes",
]


def load_splits(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        return {row["match_id"]: row["split"] for row in csv.DictReader(file)}


def normalize_label(label: str) -> str:
    if label in {"GOAL", "POSSIVEL_GOL", "POSSIBLE_GOAL"}:
        return "GOAL"
    if label in {"NO_GOAL", "NAO_GOL", "NOT_GOAL"}:
        return "NO_GOAL"
    raise ValueError(f"Rótulo de evento não suportado: {label!r}")


def load_moments(paths: list[Path]) -> dict[str, dict[str, str]]:
    moments: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                moments[row["event_id"]] = row
    return moments


def read_events(
    path: Path, splits: dict[str, str], moments: dict[str, dict[str, str]]
) -> tuple[list[dict[str, str]], Counter[str]]:
    accepted: list[dict[str, str]] = []
    rejected = Counter()
    with path.open(newline="", encoding="utf-8") as file:
        for source_row in csv.DictReader(file):
            if source_row.get("annotation_status") != "completed":
                rejected["status_pendente"] += 1
                continue
            split = splits.get(source_row["match_id"], "UNKNOWN")
            if split != "train":
                rejected[f"split_{split.lower()}"] += 1
                continue
            moment = moments.get(source_row["event_id"], {})
            accepted.append(
                {
                    "event_id": source_row["event_id"],
                    "source_file": path.as_posix(),
                    "source_interval_id": source_row.get("source_interval_id", ""),
                    "source_interval_label": source_row.get("source_interval_label", ""),
                    "match_id": source_row["match_id"],
                    "video_filename": source_row["video_filename"],
                    "split": split,
                    "event_timestamp_seconds": source_row.get(
                        "event_timestamp_seconds", ""
                    ) or moment.get("event_timestamp_seconds", ""),
                    "event_time_hhmmss": source_row.get("event_time_hhmmss", "")
                    or moment.get("event_time_hhmmss", ""),
                    "ground_truth": normalize_label(source_row["event_label"]),
                    "ball_visible": "",
                    "goal_visible": "",
                    "spatial_candidate": "",
                    "relative_motion_valid": "",
                    "identity_stable": "",
                    "center_visible_after": "",
                    "ball_at_center_after": "",
                    "players_reset": "",
                    "kickoff_observed": "",
                    "restart_time": "",
                    "scoreboard_visible": "",
                    "score_changed": "",
                    "not_evaluable_reason": "",
                    "annotation_status": "pending",
                    "notes": "Lote inicial; preencher evidências por revisão visual cega ao ground_truth.",
                }
            )
    return accepted, rejected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara o lote inicial do GOAL-EVENT-DEVSET-V1 sem acessar TEST."
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--moments", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    splits = load_splits(args.split_manifest)
    moments = load_moments(args.moments)
    rows: list[dict[str, str]] = []
    rejected = Counter()
    seen: set[str] = set()
    for path in args.events:
        source_rows, source_rejected = read_events(path, splits, moments)
        rejected.update(source_rejected)
        for row in source_rows:
            if row["event_id"] in seen:
                rejected["duplicado"] += 1
                continue
            seen.add(row["event_id"])
            rows.append(row)

    if not rows:
        raise ValueError("Nenhum evento concluído de TRAIN foi encontrado.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "GOAL-EVENT-DEVSET-V1",
        "split_policy": "somente TRAIN; VALIDATION e TEST rejeitados",
        "input_files": [path.as_posix() for path in args.events + args.moments],
        "accepted_events": len(rows),
        "accepted_by_ground_truth": dict(Counter(row["ground_truth"] for row in rows)),
        "accepted_matches": sorted({row["match_id"] for row in rows}),
        "rejected": dict(rejected),
        "annotation_status": "pending",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"eventos_train={len(rows)}")
    print(f"rótulos={summary['accepted_by_ground_truth']}")
    print(f"rejeitados={dict(rejected)}")
    print(f"csv={args.output}")
    print(f"resumo={args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
