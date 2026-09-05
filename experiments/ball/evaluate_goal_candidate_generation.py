"""Avalia a recuperação retrospectiva de candidatos por reinício no centro."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def hhmmss(seconds: float) -> str:
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator



def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia um gerador retrospectivo de candidatos usando reinícios "
            "anotados manualmente. O resultado é uma simulação, não inferência autônoma."
        )
    )
    parser.add_argument("--clip-results", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lookback-seconds",
        type=float,
        default=60.0,
        help="Duração, em segundos, da janela antes do reinício a ser revisada.",
    )
    args = parser.parse_args()

    with args.clip_results.open(newline="", encoding="utf-8") as file:
        clip_results = {row["context_id"]: row for row in csv.DictReader(file)}
    with args.contexts.open(newline="", encoding="utf-8") as file:
        contexts = [row for row in csv.DictReader(file) if row["annotation_status"] == "completed"]

    rows: list[dict[str, str]] = []
    for context in contexts:
        result = clip_results.get(context["context_id"])
        if result is None:
            raise ValueError(f"Contexto sem resultado de verificador: {context['context_id']}")
        spatial = result["spatial_candidate"] == "True"
        restart = result["restart_confirmed"] == "True"
        restart_time = context["restart_timestamp_seconds"]
        retrospective = restart and bool(restart_time)
        if spatial and retrospective:
            source = "ESPACIAL_E_REINICIO_RETROSPECTIVO"
        elif spatial:
            source = "ESPACIAL"
        elif retrospective:
            source = "REINICIO_RETROSPECTIVO"
        else:
            source = "SEM_CANDIDATO"
        candidate_generated = spatial or retrospective
        if retrospective:
            end = float(restart_time)
            start = max(0.0, end - args.lookback_seconds)
            start_value, end_value = f"{start:.3f}", f"{end:.3f}"
            start_hhmmss, end_hhmmss = hhmmss(start), hhmmss(end)
        else:
            start_value = end_value = start_hhmmss = end_hhmmss = ""
        rows.append(
            {
                "context_id": context["context_id"],
                "event_id": context["event_id"],
                "match_id": context["match_id"],
                "expected_event_label": context["expected_event_label"],
                "spatial_candidate": str(spatial),
                "restart_confirmed_manual": str(restart),
                "candidate_source": source,
                "candidate_generated": str(candidate_generated),
                "backward_window_start_seconds": start_value,
                "backward_window_end_seconds": end_value,
                "backward_window_start_hhmmss": start_hhmmss,
                "backward_window_end_hhmmss": end_hhmmss,
                "context_decision": context["context_decision"],
                "notes": "Janela retrospectiva simulada a partir de reinício anotado manualmente.",
            }
        )

    positives = [row for row in rows if row["expected_event_label"] == "POSSIVEL_GOL"]
    confirmed = [row for row in rows if row["restart_confirmed_manual"] == "True"]
    generated = [row for row in rows if row["candidate_generated"] == "True"]
    generated_positives = [row for row in positives if row["candidate_generated"] == "True"]
    generated_confirmed = [row for row in confirmed if row["candidate_generated"] == "True"]
    spatial_positive = [row for row in positives if row["spatial_candidate"] == "True"]
    spatial_confirmed = [row for row in confirmed if row["spatial_candidate"] == "True"]
    metrics = {
        "experiment_id": args.output.name,
        "evaluation_type": "simulacao_retrospetiva_com_reinicio_anotado_manual",
        "contexts": len(rows),
        "lookback_seconds": args.lookback_seconds,
        "spatial_candidates": sum(row["spatial_candidate"] == "True" for row in rows),
        "hybrid_candidates": len(generated),
        "positive_events": len(positives),
        "spatial_candidate_recall_on_positive_events": rate(len(spatial_positive), len(positives)),
        "hybrid_candidate_recall_on_positive_events": rate(len(generated_positives), len(positives)),
        "restart_confirmed_events": len(confirmed),
        "spatial_candidate_recall_on_restart_confirmed": rate(len(spatial_confirmed), len(confirmed)),
        "hybrid_candidate_recall_on_restart_confirmed": rate(len(generated_confirmed), len(confirmed)),
        "recovered_only_by_restart": sum(row["candidate_source"] == "REINICIO_RETROSPECTIVO" for row in rows),
        "warning": "O reinicio e manual neste piloto; nao interpretar como desempenho de detector automatico.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "retrospective_candidate_windows.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
