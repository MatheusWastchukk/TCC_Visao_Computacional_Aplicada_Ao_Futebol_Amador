"""Estratifica o ranking TRAIN usando as notas de goleira secundária por partida."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
NOTES = ROOT / "data/Videos_Edited/Contagem Gols por Jogo.txt"
TRACE = EXP / "selection_trace.csv"
MANIFEST = EXP / "case_manifest.csv"
GAPS = (20, 30, 40, 50, 60, 80)
GAP_SECONDS = 45


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_annotations() -> dict[str, dict[str, str]]:
    text = NOTES.read_text(encoding="utf-8-sig")
    blocks: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"\s*Match\s*_?\s*(\d+)\s*:?\s*$", line, re.IGNORECASE)
        if match:
            current = f"match_{int(match.group(1)):02d}"
            blocks[current] = []
        elif current:
            blocks[current].append((line_number, line.strip()))

    annotations: dict[str, dict[str, str]] = {}
    for match_id, lines in blocks.items():
        body = " ".join(line for _, line in lines).casefold()
        if "não tem gols secundários" in body or "nao tem gols secundarios" in body:
            status = "NO_EXPLICIT"
            evidence = "Não tem gols secundários"
        elif "parecem gol secundários" in body or "parecem gols secundários" in body or "parecem gol secundarios" in body:
            status = "UNCERTAIN"
            evidence = "Há estruturas que parecem gols secundários"
        elif "tem gols secundários" in body or "tem diversos gols secundários" in body:
            status = "YES_EXPLICIT"
            evidence = "Anotação textual explícita de gols secundários"
        else:
            status = "NOT_MENTIONED"
            evidence = "O arquivo não registra presença nem ausência"
        evidence_line = next((number for number, line in lines if line and any(
            token in line.casefold() for token in ("gols secund", "gol secund", "estruturas que parecem")
        )), "")
        source_excerpt = next((line for _, line in lines if line and any(
            token in line.casefold() for token in ("gols secund", "gol secund", "estruturas que parecem")
        )), "")
        annotations[match_id] = {
            "match_id": match_id,
            "secondary_goal_match_status": status,
            "source_file": "data/Videos_Edited/Contagem Gols por Jogo.txt",
            "source_line": evidence_line,
            "source_excerpt": source_excerpt,
            "source_evidence_summary": evidence,
            "annotation_scope": "match-level; não informa se a estrutura aparece na janela de cada candidato",
        }
    return annotations


def merge_intervals(rows: list[dict[str, str]]) -> tuple[int, float]:
    intervals = sorted((float(r["delivery_start_seconds"]), float(r["delivery_end_seconds"])) for r in rows)
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return len(merged), sum(end - start for start, end in merged)


def main() -> int:
    annotations = read_annotations()
    trace = read_csv(TRACE)
    manifest = read_csv(MANIFEST)
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    goals_by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        by_match[row["match_id"]].append(row)
    for row in manifest:
        goals_by_match[row["match_id"]].append(row)

    train_matches = sorted(by_match)
    missing = sorted(set(train_matches) - set(annotations))
    if missing:
        raise ValueError(f"Partidas TRAIN ausentes das notas: {missing}")

    annotation_rows = [annotations[match] for match in train_matches]
    with (EXP / "secondary_goal_match_annotations_from_game_notes.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)

    per_match_results: list[dict[str, object]] = []
    for match in train_matches:
        status = annotations[match]["secondary_goal_match_status"]
        candidates = sorted(by_match[match], key=lambda r: (-float(r["temporal_score_lomo"]), r["group_uid"]))
        duration_seconds = max(float(goal["video_duration_seconds"]) for goal in goals_by_match[match])
        for k in GAPS:
            selected: list[dict[str, str]] = []
            for candidate in candidates:
                center = float(candidate["best_timestamp_seconds"])
                if all(abs(center - float(previous["best_timestamp_seconds"])) >= GAP_SECONDS for previous in selected):
                    selected.append(candidate)
                if len(selected) == k:
                    break
            covered = [goal for goal in goals_by_match[match] if any(
                float(candidate["delivery_start_seconds"]) <= float(goal["goal_timestamp_seconds"]) <= float(candidate["delivery_end_seconds"])
                for candidate in selected
            )]
            recoverable = [goal for goal in goals_by_match[match] if goal["pool_recoverable"].casefold() == "true"]
            covered_recoverable = [goal for goal in recoverable if goal in covered]
            segments, seconds = merge_intervals(selected)
            per_match_results.append({
                "secondary_goal_match_status": status,
                "match_id": match,
                "k": k,
                "gap_seconds": GAP_SECONDS,
                "catalog_goals": len(goals_by_match[match]),
                "covered_catalog_goals": len(covered),
                "pool_recoverable_goals": len(recoverable),
                "covered_pool_recoverable_goals": len(covered_recoverable),
                "pool_candidate_groups": len(candidates),
                "pool_groups_per_video_minute": round(len(candidates) / (duration_seconds / 60), 4),
                "selected_candidates": len(selected),
                "merged_delivery_segments": segments,
                "unique_delivery_minutes": round(seconds / 60, 3),
            })

    summary_rows: list[dict[str, object]] = []
    comparable_statuses = ("YES_EXPLICIT", "NO_EXPLICIT")
    for status in comparable_statuses:
        matches = [m for m in train_matches if annotations[m]["secondary_goal_match_status"] == status]
        for k in GAPS:
            rows = [r for r in per_match_results if r["match_id"] in matches and r["k"] == k]
            catalog = sum(int(r["catalog_goals"]) for r in rows)
            covered = sum(int(r["covered_catalog_goals"]) for r in rows)
            pool = sum(int(r["pool_recoverable_goals"]) for r in rows)
            pool_covered = sum(int(r["covered_pool_recoverable_goals"]) for r in rows)
            duration_minutes = sum(max(float(g["video_duration_seconds"]) for g in goals_by_match[m]) / 60 for m in matches)
            candidates_pool = sum(int(r["pool_candidate_groups"]) for r in rows)
            summary_rows.append({
                "secondary_goal_match_status": status,
                "k": k,
                "gap_seconds": GAP_SECONDS,
                "matches": len(matches),
                "catalog_goals": catalog,
                "covered_catalog_goals": covered,
                "catalog_recall": round(covered / catalog, 6) if catalog else "",
                "pool_recoverable_goals": pool,
                "covered_pool_recoverable_goals": pool_covered,
                "conditional_pool_recall": round(pool_covered / pool, 6) if pool else "",
                "pool_candidate_groups": candidates_pool,
                "pool_groups_per_video_minute": round(candidates_pool / duration_minutes, 4) if duration_minutes else "",
                "selected_candidates": sum(int(r["selected_candidates"]) for r in rows),
                "merged_delivery_segments": sum(int(r["merged_delivery_segments"]) for r in rows),
                "unique_delivery_minutes_total": round(sum(float(r["unique_delivery_minutes"]) for r in rows), 3),
                "unique_delivery_minutes_per_match": round(sum(float(r["unique_delivery_minutes"]) for r in rows) / len(matches), 3) if matches else "",
            })

    for filename, rows in (("secondary_goal_match_stratification_by_match.csv", per_match_results),
                           ("secondary_goal_match_stratification.csv", summary_rows)):
        with (EXP / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    counts = {status: sum(annotations[m]["secondary_goal_match_status"] == status for m in train_matches)
              for status in ("YES_EXPLICIT", "NO_EXPLICIT", "UNCERTAIN", "NOT_MENTIONED")}
    md = [
        "# Estratificação TRAIN por anotação textual de goleira secundária",
        "",
        "## Fonte e regra de rótulo",
        "",
        "Fonte: `data/Videos_Edited/Contagem Gols por Jogo.txt`. Foram usados somente rótulos no nível da partida declarados no texto. Menções explícitas positivas e negativas foram preservadas; ‘parecem estruturas’ foi tratado como incerto; ausência de menção não foi convertida em negativo.",
        "",
        f"Nas 25 partidas TRAIN: {counts['YES_EXPLICIT']} positivas explícitas, {counts['NO_EXPLICIT']} negativas explícitas, {counts['UNCERTAIN']} incerta(s) e {counts['NOT_MENTIONED']} sem anotação. A comparação quantitativa inclui apenas as {counts['YES_EXPLICIT'] + counts['NO_EXPLICIT']} partidas com rótulo explícito.",
        "",
        "## Curvas de cobertura e custo (gap45 congelado)",
        "",
        "Scores RF LOMO, entregas 20+40 s e gap45 foram mantidos; variou somente K. A cobertura usa timestamps do catálogo e cada gol é contado uma vez. Minutos únicos são tempo de vídeo entregue, não tempo humano.",
        "",
        "| Estrato | K | Partidas | Gols cobertos / catálogo | Recall condicional do pool | Grupos pool/min de vídeo | Candidatos | Segmentos | Minutos únicos/jogo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['secondary_goal_match_status']} | {row['k']} | {row['matches']} | "
            f"{row['covered_catalog_goals']}/{row['catalog_goals']} ({float(row['catalog_recall']):.1%}) | "
            f"{row['covered_pool_recoverable_goals']}/{row['pool_recoverable_goals']} ({float(row['conditional_pool_recall']):.1%}) | "
            f"{row['pool_groups_per_video_minute']} | {row['selected_candidates']} | "
            f"{row['merged_delivery_segments']} | {row['unique_delivery_minutes_per_match']} |"
        )
    md.extend([
        "",
        "## Limites de interpretação",
        "",
        "Este é um contraste observacional pequeno, por partida; não estima causalidade da segunda goleira. Os grupos podem diferir em câmera, qualidade, duração e número de gols. Rótulos incertos e sem menção foram excluídos da comparação binária, mas permanecem no arquivo de anotações.",
        "",
        "A fonte não informa se a estrutura secundária aparece em cada janela/candidato. Portanto, a variável `secondary_goal_candidate`, a contagem de falsos Top-20 associados a ela e comparações visuais pareadas continuam pendentes. Não foram inferidos rótulos de candidato nem usados overlays. VALIDATION e TEST não foram acessados.",
        "",
        "Arquivos: `secondary_goal_match_annotations_from_game_notes.csv`, `secondary_goal_match_stratification.csv` e `secondary_goal_match_stratification_by_match.csv`.",
    ])
    (EXP / "secondary_goal_match_stratification.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"matches={len(train_matches)} statuses={counts} output_rows={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
