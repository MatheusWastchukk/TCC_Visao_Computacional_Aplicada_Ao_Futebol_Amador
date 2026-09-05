"""Prepara comparações antes/depois para evidência opcional de placar."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "scoreboard_id", "context_id", "event_id", "match_id", "video_filename",
    "event_timestamp_seconds", "before_timestamp_seconds", "after_timestamp_seconds",
    "before_hhmmss", "after_hhmmss", "scoreboard_available", "scoreboard_roi_json",
    "score_changed", "annotation_status", "notes",
]


def hhmmss(seconds: float) -> str:
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara piloto de evidência por mudança de placar.")
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="CSV do split por partida. Quando informado junto com --split, impede misturar conjuntos.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="Conjunto aceito no experimento. Requer --split-manifest.",
    )
    parser.add_argument("--before-seconds", type=float, default=10.0)
    parser.add_argument("--fallback-after-seconds", type=float, default=90.0)
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split-manifest e --split devem ser informados juntos.")

    with args.contexts.open(newline="", encoding="utf-8") as file:
        contexts = [row for row in csv.DictReader(file) if row["annotation_status"] == "completed"]
    if args.split_manifest:
        with args.split_manifest.open(newline="", encoding="utf-8") as file:
            split_by_match = {row["match_id"]: row["split"] for row in csv.DictReader(file)}
        missing = sorted({row["match_id"] for row in contexts if row["match_id"] not in split_by_match})
        if missing:
            raise ValueError(f"Partidas sem split definido: {', '.join(missing)}")
        contexts = [row for row in contexts if split_by_match[row["match_id"]] == args.split]
    rows = []
    for index, context in enumerate(contexts, 1):
        event_time = float(context["event_timestamp_seconds"])
        after = float(context["restart_timestamp_seconds"]) if context["restart_timestamp_seconds"] else event_time + args.fallback_after_seconds
        before = max(0.0, event_time - args.before_seconds)
        rows.append({
            "scoreboard_id": f"scoreboard_context_{index:03d}", "context_id": context["context_id"],
            "event_id": context["event_id"], "match_id": context["match_id"], "video_filename": context["video_filename"],
            "event_timestamp_seconds": f"{event_time:.3f}", "before_timestamp_seconds": f"{before:.3f}",
            "after_timestamp_seconds": f"{after:.3f}", "before_hhmmss": hhmmss(before), "after_hhmmss": hhmmss(after),
            "scoreboard_available": "", "scoreboard_roi_json": "", "score_changed": "", "annotation_status": "pending",
            "notes": "Piloto manual: placar é evidência opcional e não substitui a verificação do evento.",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"contextos={len(rows)}\ncsv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
