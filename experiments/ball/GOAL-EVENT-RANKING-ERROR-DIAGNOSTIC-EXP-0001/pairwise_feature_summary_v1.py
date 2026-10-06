"""Resume diferenças de features em pares sem confirmar rótulos negativos."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
INPUT = EXP / "pairwise_rank_diagnostic.csv"


def main() -> int:
    with INPUT.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    columns = [key for key in rows[0] if key.startswith("delta_candidate_minus_goal_")]
    summaries = []
    for column in columns:
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                continue
        if values:
            summaries.append({
                "feature": column.removeprefix("delta_candidate_minus_goal_"),
                "paired_rows": len(values),
                "candidate_higher_fraction": round(sum(v > 0 for v in values) / len(values), 4),
                "goal_higher_fraction": round(sum(v < 0 for v in values) / len(values), 4),
                "tie_fraction": round(sum(v == 0 for v in values) / len(values), 4),
                "median_candidate_minus_goal": round(statistics.median(values), 6),
                "mean_candidate_minus_goal": round(statistics.mean(values), 6),
            })
    summaries.sort(key=lambda r: (abs(float(r["median_candidate_minus_goal"])), str(r["feature"])), reverse=True)
    with (EXP / "pairwise_feature_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    statuses = {}
    for row in rows:
        status = row.get("pair_label_status", "")
        statuses[status] = statuses.get(status, 0) + 1
    md = [
        "# Resumo pareado de features e ranking — TRAIN",
        "",
        f"Foram resumidos {len(rows)} pares associados a {len(set(r['goal_catalog_id'] for r in rows))} gols catalogados abaixo do Top-60. Status dos comparadores: `{statuses}`.",
        "",
        "O grupo do timestamp catalogado é referência positiva. Os candidatos comparadores não cobrem timestamp de gol catalogado e têm rank superior na mesma partida, mas isso **não confirma que sejam falsos**; o catálogo pode ser incompleto e a confirmação humana não foi feita. Os valores são diferenças descritivas, sem treinamento ou inferência causal.",
        "",
        "Delta positivo significa valor maior no candidato comparador; negativo significa valor maior no grupo que cobre o gol catalogado. Features com escalas diferentes não devem ser comparadas pela magnitude entre si.",
        "",
        "| Feature | Pares | Fração candidato maior | Mediana (candidato − gol) |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        md.append(f"| {row['feature']} | {row['paired_rows']} | {row['candidate_higher_fraction']:.1%} | {row['median_candidate_minus_goal']} |")
    md.extend([
        "",
        "**Limite:** esta tabela não determina quais features distinguem gol de não-gol e não justifica retreinar ou ajustar o Random Forest. Para isso, os comparadores precisam de rótulos confiáveis. VALIDATION e TEST permaneceram fechados.",
        "",
        "Dados: `pairwise_rank_diagnostic.csv`; resumo: `pairwise_feature_summary.csv`.",
    ])
    (EXP / "pairwise_feature_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"pair_rows={len(rows)} goals={len(set(r['goal_catalog_id'] for r in rows))} features={len(summaries)} statuses={statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
