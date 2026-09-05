"""Resume o valor da evidência manual de mudança de placar."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia placar como evidência opcional de confirmação de evento.")
    parser.add_argument("--scoreboard", type=Path, required=True)
    parser.add_argument("--clip-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.scoreboard.open(newline="", encoding="utf-8") as file:
        scoreboard = [row for row in csv.DictReader(file) if row["annotation_status"] == "completed"]
    with args.clip_results.open(newline="", encoding="utf-8") as file:
        clip = {row["context_id"]: row for row in csv.DictReader(file)}
    rows = []
    for row in scoreboard:
        context = clip.get(row["context_id"])
        if context is None:
            raise ValueError(f"Contexto sem resultado de clipe: {row['context_id']}")
        changed = row["score_changed"] == "True"
        available = row["scoreboard_available"] == "True"
        if changed:
            evidence = "SUPORTA_GOAL"
        elif available:
            evidence = "SEM_SUPORTE_POR_PLACAR"
        else:
            evidence = "PLACAR_INDISPONIVEL"
        rows.append({
            "scoreboard_id": row["scoreboard_id"], "context_id": row["context_id"], "event_id": row["event_id"],
            "match_id": row["match_id"], "scoreboard_available": str(available), "score_changed": str(changed),
            "restart_confirmed": context["restart_confirmed"], "spatial_candidate": context["spatial_candidate"],
            "clip_decision": context["final_decision"], "scoreboard_evidence": evidence,
        })
    available = [row for row in rows if row["scoreboard_available"] == "True"]
    changed = [row for row in rows if row["score_changed"] == "True"]
    metrics = {
        "experiment_id": args.output.name, "contexts": len(rows), "scoreboard_available": len(available),
        "scoreboard_changed": len(changed), "supporting_evidence_rate_when_available": len(changed) / len(available) if available else None,
        "changed_with_restart_confirmed": sum(row["restart_confirmed"] == "True" for row in changed),
        "changed_with_possible_goal": sum(row["clip_decision"] == "POSSIBLE_GOAL" for row in changed),
        "interpretation": "Mudança de placar apoia a confirmação; ausência de mudança não rejeita o evento.",
        "warning": "Revisão manual pequena; não é OCR nem detector automático de placar.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "scoreboard_context_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
