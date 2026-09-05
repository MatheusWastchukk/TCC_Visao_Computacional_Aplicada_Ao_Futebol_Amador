"""Prepara contexto pós-evento a partir do catálogo de gols principais."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "context_id", "event_id", "expected_event_label", "match_id", "video_filename",
    "event_timestamp_seconds", "search_start_seconds", "search_end_seconds",
    "search_start_hhmmss", "search_end_hhmmss", "center_visible", "ball_at_center",
    "players_reset", "kickoff_observed", "score_changed", "camera_lost", "end_of_half",
    "restart_timestamp_seconds", "context_decision", "annotation_status", "notes",
]


def hhmmss(seconds: float) -> str:
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_video(videos_dir: Path, match_id: str) -> Path:
    numeric = match_id.split("_")[-1]
    candidates = sorted(videos_dir.glob(f"Match_{numeric}_*.mp4")) + sorted(videos_dir.glob(f"Match_{numeric}.mp4"))
    if len(candidates) != 1:
        raise ValueError(f"Não foi possível resolver exatamente um vídeo para {match_id}: {candidates}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara GOAL-CONTEXT-EXP-0002 a partir de gols principais catalogados.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-after", type=float, default=10.0)
    parser.add_argument("--window", type=float, default=120.0)
    args = parser.parse_args()
    with args.catalog.open(newline="", encoding="utf-8") as file:
        catalog = [row for row in csv.DictReader(file) if row["label"] == "main_goal"]
    rows: list[dict[str, str]] = []
    for index, item in enumerate(catalog, 1):
        event_time = float(item["timestamp_seconds"])
        start = event_time + args.start_after
        end = start + args.window
        video = resolve_video(args.videos_dir, item["match_id"])
        rows.append(
            {
                "context_id": f"goal_context_catalog_{index:03d}",
                "event_id": f"catalog_{item['match_id']}_{index:03d}",
                "expected_event_label": "GOAL_MAIN_CATALOG",
                "match_id": item["match_id"],
                "video_filename": video.name,
                "event_timestamp_seconds": f"{event_time:.3f}",
                "search_start_seconds": f"{start:.3f}",
                "search_end_seconds": f"{end:.3f}",
                "search_start_hhmmss": hhmmss(start),
                "search_end_hhmmss": hhmmss(end),
                "center_visible": "", "ball_at_center": "", "players_reset": "",
                "kickoff_observed": "", "score_changed": "", "camera_lost": "", "end_of_half": "",
                "restart_timestamp_seconds": "", "context_decision": "", "annotation_status": "pending",
                "notes": "Gol principal de catálogo; confirmar somente o contexto visual pós-evento.",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"contextos={len(rows)}\ncsv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
