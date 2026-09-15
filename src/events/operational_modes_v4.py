"""Seleciona e une candidatos conforme os modos operacionais V4 congelados."""
from __future__ import annotations

from collections.abc import Iterable


REQUIRED_FIELDS = {
    "event_id",
    "match_id",
    "video_filename",
    "event_timestamp_seconds",
    "rank_score",
    "fusion_score_goal",
    "delivery_start_seconds",
    "delivery_end_seconds",
}

MODES = {
    "alta_cobertura": "Top 60 pelo ranking temporal.",
    "cobertura_reduzida": "Top 30 pela fusão e 15 complementos temporais.",
    "triagem_rapida": "Top 30 pela fusão temporal e visual.",
}


def format_time(seconds: float) -> str:
    """Formata segundos como horas, minutos e segundos."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def validate_candidates(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Valida um pool pontuado de uma única partida, sem usar rótulos de gol."""
    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("O pool de candidatos está vazio.")
    if missing := REQUIRED_FIELDS - set(candidates[0]):
        raise ValueError(f"Candidatos sem colunas obrigatórias: {sorted(missing)}")
    if any(set(candidate) < REQUIRED_FIELDS for candidate in candidates):
        raise ValueError("Há candidatos com colunas incompatíveis.")
    if len({candidate["match_id"] for candidate in candidates}) != 1:
        raise ValueError("A análise operacional aceita uma única partida por execução.")
    if len({candidate["video_filename"] for candidate in candidates}) != 1:
        raise ValueError("Os candidatos devem apontar para um único vídeo.")
    if len({candidate["event_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Os identificadores de candidatos devem ser únicos.")
    for candidate in candidates:
        start, end = float(candidate["delivery_start_seconds"]), float(candidate["delivery_end_seconds"])
        if start < 0 or end <= start:
            raise ValueError("Os intervalos de entrega devem ser positivos e ordenados.")
    return candidates


def select_candidates(rows: Iterable[dict[str, str]], mode: str) -> list[dict[str, str]]:
    """Aplica estritamente um dos modos V4, sem recalibração de escores."""
    candidates = validate_candidates(rows)
    if mode not in MODES:
        raise ValueError(f"Modo desconhecido: {mode}. Opções: {sorted(MODES)}")
    temporal = sorted(candidates, key=lambda row: float(row["rank_score"]), reverse=True)
    fusion = sorted(candidates, key=lambda row: float(row["fusion_score_goal"]), reverse=True)
    if mode == "alta_cobertura":
        return temporal[:60]
    if mode == "triagem_rapida":
        return fusion[:30]
    selected = fusion[:30]
    selected_ids = {row["event_id"] for row in selected}
    for row in temporal:
        if row["event_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["event_id"])
            if len(selected) == 45:
                break
    return selected


def merge_candidates(rows: Iterable[dict[str, str]], mode: str) -> list[dict[str, object]]:
    """Une intervalos sobrepostos e preserva os candidatos que formam cada segmento."""
    selected = select_candidates(rows, mode)
    ordered = sorted(selected, key=lambda row: float(row["delivery_start_seconds"]))
    merged: list[dict[str, object]] = []
    for candidate in ordered:
        start, end = float(candidate["delivery_start_seconds"]), float(candidate["delivery_end_seconds"])
        if not merged or start > float(merged[-1]["end_seconds"]):
            merged.append({"start_seconds": start, "end_seconds": end, "members": [candidate]})
        else:
            merged[-1]["end_seconds"] = max(float(merged[-1]["end_seconds"]), end)
            members = merged[-1]["members"]
            assert isinstance(members, list)
            members.append(candidate)
    segments: list[dict[str, object]] = []
    for index, item in enumerate(merged, start=1):
        members = item["members"]
        assert isinstance(members, list)
        strongest = max(members, key=lambda row: float(row["fusion_score_goal"]))
        start, end = float(item["start_seconds"]), float(item["end_seconds"])
        segments.append({
            "segment_id": f"segment_{index:03d}",
            "label": "POSSIBLE_GOAL",
            "start_seconds": start,
            "end_seconds": end,
            "start_hhmmss": format_time(start),
            "end_hhmmss": format_time(end),
            "duration_seconds": end - start,
            "representative_event_id": strongest["event_id"],
            "representative_timestamp_seconds": float(strongest["event_timestamp_seconds"]),
            "representative_time_hhmmss": format_time(float(strongest["event_timestamp_seconds"])),
            "rank_score": float(strongest["rank_score"]),
            "fusion_score_goal": float(strongest["fusion_score_goal"]),
            "member_event_ids": [row["event_id"] for row in members],
            "member_count": len(members),
        })
    return segments


def analyze_match(rows: Iterable[dict[str, str]], mode: str) -> dict[str, object]:
    """Gera a saída operacional de uma partida a partir de seu pool já pontuado."""
    candidates = validate_candidates(rows)
    selected = select_candidates(candidates, mode)
    segments = merge_candidates(candidates, mode)
    return {
        "schema_version": "goal-event-operational-v4",
        "match_id": candidates[0]["match_id"],
        "video_filename": candidates[0]["video_filename"],
        "mode": mode,
        "mode_description": MODES[mode],
        "label_policy": "POSSIBLE_GOAL; revisão humana obrigatória; não confirma gol automaticamente.",
        "selected_candidates": selected,
        "segments": segments,
        "summary": {
            "selected_candidates": len(selected),
            "segments_after_merge": len(segments),
            "unique_review_seconds": sum(float(segment["duration_seconds"]) for segment in segments),
        },
    }
