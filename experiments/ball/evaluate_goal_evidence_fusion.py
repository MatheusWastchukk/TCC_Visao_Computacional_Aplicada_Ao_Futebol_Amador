"""Funde candidato espacial, reinício e placar em decisão experimental de evento."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia a fusão de evidências espaciais e contextuais de gol.")
    parser.add_argument("--clip-results", type=Path, required=True)
    parser.add_argument("--scoreboard-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.clip_results.open(newline="", encoding="utf-8") as file:
        clip_results = {row["context_id"]: row for row in csv.DictReader(file)}
    with args.scoreboard_results.open(newline="", encoding="utf-8") as file:
        scoreboard = {row["context_id"]: row for row in csv.DictReader(file)}
    missing_contexts = sorted(set(scoreboard) - set(clip_results))
    if missing_contexts:
        raise ValueError(
            "Contextos de placar sem resultado de clipe: "
            + ", ".join(missing_contexts)
        )
    rows = []
    for context_id in sorted(scoreboard):
        clip = clip_results[context_id]
        score = scoreboard[context_id]
        spatial = clip["spatial_candidate"] == "True"
        restart = clip["restart_confirmed"] == "True"
        score_changed = score["score_changed"] == "True"
        if restart and score_changed:
            decision = "GOAL_WITH_CONTEXT_EVIDENCE"
        elif restart and spatial:
            decision = "POSSIBLE_GOAL"
        elif restart:
            decision = "MISSED_SPATIAL_CANDIDATE"
        elif spatial:
            decision = "NO_GOAL_OR_INVALIDATED"
        else:
            decision = "NOT_EVALUABLE"
        rows.append({
            "context_id": context_id, "event_id": clip["event_id"], "match_id": clip["match_id"],
            "expected_event_label": clip["expected_event_label"], "spatial_candidate": str(spatial),
            "restart_confirmed": str(restart), "score_changed": str(score_changed),
            "scoreboard_evidence": score["scoreboard_evidence"], "final_decision": decision,
        })
    supported = [row for row in rows if row["final_decision"] in {"GOAL_WITH_CONTEXT_EVIDENCE", "POSSIBLE_GOAL"}]
    metrics = {
        "experiment_id": args.output.name, "contexts": len(rows),
        "goal_with_context_evidence": sum(row["final_decision"] == "GOAL_WITH_CONTEXT_EVIDENCE" for row in rows),
        "possible_goal": sum(row["final_decision"] == "POSSIBLE_GOAL" for row in rows),
        "supported_goal_events": len(supported),
        "missed_spatial_candidate": sum(row["final_decision"] == "MISSED_SPATIAL_CANDIDATE" for row in rows),
        "rejected_or_invalidated": sum(row["final_decision"] == "NO_GOAL_OR_INVALIDATED" for row in rows),
        "rule": "placar_alterado_e_reinicio > reinicio_e_candidato_espacial > demais_estados",
        "warning": "Fusão experimental em revisão manual pequena; não equivale a confirmação oficial de gol.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "goal_evidence_fusion_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
