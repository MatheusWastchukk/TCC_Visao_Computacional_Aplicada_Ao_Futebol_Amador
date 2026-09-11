"""Prepara candidatos inéditos a partir de ranking LOMO, somente TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EVENT_FIELDS = [
    "event_id", "source_file", "source_interval_id", "source_interval_label", "match_id",
    "video_filename", "split", "event_timestamp_seconds", "event_time_hhmmss", "ground_truth",
    "ball_visible", "goal_visible", "spatial_candidate", "relative_motion_valid", "identity_stable",
    "center_visible_after", "ball_at_center_after", "players_reset", "kickoff_observed", "restart_time",
    "scoreboard_visible", "score_changed", "not_evaluable_reason", "annotation_status", "notes",
    "candidate_score", "candidate_hit_frames", "candidate_start_seconds", "candidate_end_seconds", "rank_score",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara candidatos alternativos TRAIN pelo ranking leave-one-match-out."
    )
    parser.add_argument("--ranked-groups", type=Path, required=True)
    parser.add_argument("--existing-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates-per-match", type=int, default=4)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    parser.add_argument("--candidate-version", default="v2")
    parser.add_argument("--model", help="Filtra uma coluna model do arquivo de ranking.")
    args = parser.parse_args()
    if args.max_candidates_per_match < 1 or args.min_separation_seconds < 0:
        raise ValueError("Os limites de seleção devem ser válidos.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}.")

    groups, group_fields = read_csv(args.ranked_groups)
    required_groups = {"match_id", "best_timestamp_seconds", "candidate_start_seconds", "candidate_end_seconds", "rank_score"}
    if missing := required_groups - set(group_fields):
        raise ValueError(f"Ranking sem colunas obrigatórias: {sorted(missing)}")

    existing_times: dict[str, list[float]] = defaultdict(list)
    video_by_match: dict[str, str] = {}
    for path in args.existing_candidates:
        existing, existing_fields = read_csv(path)
        required_existing = {"match_id", "video_filename", "split", "event_timestamp_seconds"}
        if missing := required_existing - set(existing_fields):
            raise ValueError(f"Candidatos existentes sem colunas em {path}: {sorted(missing)}")
        if any(row["split"] != "train" for row in existing):
            raise AssertionError("A preparação aceita somente candidatos TRAIN; TEST permanece fechado.")
        for row in existing:
            existing_times[row["match_id"]].append(float(row["event_timestamp_seconds"]))
            video_by_match[row["match_id"]] = row["video_filename"]
    groups_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in groups:
        if args.model and row.get("model") != args.model:
            continue
        if row["match_id"] not in video_by_match:
            raise AssertionError(f"Grupo sem candidato TRAIN existente: {row['match_id']}")
        groups_by_match[row["match_id"]].append(row)

    output_rows: list[dict[str, str]] = []
    selected_by_match: dict[str, int] = {}
    for match_id in sorted(groups_by_match):
        selected: list[dict[str, str]] = []
        blocked = existing_times[match_id].copy()
        for group in sorted(groups_by_match[match_id], key=lambda row: float(row["rank_score"]), reverse=True):
            timestamp = float(group["best_timestamp_seconds"])
            if any(abs(timestamp - other) < args.min_separation_seconds for other in blocked):
                continue
            selected.append(group)
            blocked.append(timestamp)
            if len(selected) >= args.max_candidates_per_match:
                break
        selected_by_match[match_id] = len(selected)
        for group in selected:
            timestamp = float(group["best_timestamp_seconds"])
            output_rows.append({
                "event_id": f"{match_id}__ranker_candidate_{args.candidate_version}_{len(output_rows) + 1:04d}",
                "source_file": "evaluate_goal_event_group_ranker_lomo_v1.py",
                "source_interval_id": group["group_uid"],
                "source_interval_label": "ranking_lomo_grupo_inedito",
                "match_id": match_id,
                "video_filename": video_by_match[match_id],
                "split": "train",
                "event_timestamp_seconds": f"{timestamp:.3f}",
                "event_time_hhmmss": format_time(timestamp),
                "ground_truth": "", "ball_visible": "", "goal_visible": "", "spatial_candidate": "True",
                "relative_motion_valid": "", "identity_stable": "", "center_visible_after": "",
                "ball_at_center_after": "", "players_reset": "", "kickoff_observed": "", "restart_time": "",
                "scoreboard_visible": "", "score_changed": "", "not_evaluable_reason": "", "annotation_status": "pending",
                "notes": "Candidato alternativo LOMO; separado dos clipes V1 já revisados.",
                "candidate_score": "", "candidate_hit_frames": "",
                "candidate_start_seconds": group["candidate_start_seconds"],
                "candidate_end_seconds": group["candidate_end_seconds"],
                "rank_score": group["rank_score"],
            })
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "candidate_events_v2.csv", output_rows)
    summary = {
        "experiment_id": "GOAL-EVENT-RANKER-CANDIDATES-V2",
        "scope": "somente TRAIN; ranking leave-one-match-out",
        "candidate_events": len(output_rows),
        "max_candidates_per_match": args.max_candidates_per_match,
        "selected_by_match": selected_by_match,
        "note": "Candidatos inéditos para revisão humana. Não representa avaliação final nem confirmação online.",
    }
    (args.output / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
