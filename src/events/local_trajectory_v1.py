"""Extrai uma trajetória local consistente como evidência experimental.

O módulo não participa da seleção V4 congelada. Ele escolhe, por programação
dinâmica, uma sequência plausível entre detecções próximas no tempo e retorna
atributos diagnósticos para futura avaliação controlada.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import exp
from collections.abc import Iterable


@dataclass(frozen=True)
class BallObservation:
    """Centro e confiança de uma detecção de bola em um quadro."""

    timestamp_seconds: float
    x: float
    y: float
    confidence: float
    reference_scale: float
    goal_x: float = 0.0
    goal_y: float = 0.0


@dataclass(frozen=True)
class LocalTrajectory:
    """Resumo de uma cadeia de detecções localmente coerente."""

    observations: tuple[BallObservation, ...]
    path_score: float
    mean_normalized_step: float
    continuity: float


def observations_from_frame_rows(rows: Iterable[dict[str, str]]) -> list[BallObservation]:
    """Converte o CSV de hits do minerador em observações da trajetória.

    O adaptador consome somente colunas já produzidas pelo minerador. Assim, o
    experimento pode ser executado sobre saídas existentes sem reexecutar os
    detectores nem consultar rótulos de referência.
    """
    observations: list[BallObservation] = []
    for row in rows:
        ball = [float(value) for value in json.loads(row["ball_box_json"])]
        goal = [float(value) for value in json.loads(row["goal_box_json"])]
        if len(ball) != 4 or len(goal) != 4:
            raise ValueError("Cada caixa deve conter quatro coordenadas.")
        scale = ((goal[2] - goal[0]) ** 2 + (goal[3] - goal[1]) ** 2) ** 0.5
        observations.append(BallObservation(
            timestamp_seconds=float(row["timestamp_seconds"]),
            x=(ball[0] + ball[2]) / 2.0,
            y=(ball[1] + ball[3]) / 2.0,
            confidence=float(row["ball_confidence"]),
            reference_scale=max(1.0, scale),
            goal_x=(goal[0] + goal[2]) / 2.0,
            goal_y=(goal[1] + goal[3]) / 2.0,
        ))
    return observations


def normalized_step(first: BallObservation, second: BallObservation) -> float:
    """Distância entre centros, normalizada pela escala local da trave."""
    scale = max(1.0, first.reference_scale, second.reference_scale)
    return (((second.x - first.x) ** 2 + (second.y - first.y) ** 2) ** 0.5) / scale


def best_local_trajectory(
    observations: Iterable[BallObservation], *, max_gap_seconds: float = 1.0,
    transition_penalty: float = 2.0,
) -> LocalTrajectory:
    """Escolhe a cadeia de maior evidência, penalizando saltos entre quadros.

    A pontuação combina confiança da detecção e deslocamento normalizado. Não
    é probabilidade de gol e não deve ser usada como limiar operacional antes
    de uma avaliação independente.
    """
    ordered = sorted(observations, key=lambda item: item.timestamp_seconds)
    if not ordered:
        return LocalTrajectory((), 0.0, 0.0, 0.0)
    scores = [max(0.0, item.confidence) for item in ordered]
    parents = [-1] * len(ordered)
    for current, item in enumerate(ordered):
        for previous in range(current):
            before = ordered[previous]
            gap = item.timestamp_seconds - before.timestamp_seconds
            if gap <= 0 or gap > max_gap_seconds:
                continue
            transition = normalized_step(before, item)
            candidate = scores[previous] + max(0.0, item.confidence) - transition_penalty * transition
            if candidate > scores[current]:
                scores[current] = candidate
                parents[current] = previous
    endpoint = max(range(len(ordered)), key=scores.__getitem__)
    path_score = scores[endpoint]
    indexes: list[int] = []
    while endpoint >= 0:
        indexes.append(endpoint)
        endpoint = parents[endpoint]
    path = tuple(ordered[index] for index in reversed(indexes))
    steps = [normalized_step(first, second) for first, second in zip(path, path[1:])]
    mean_step = sum(steps) / len(steps) if steps else 0.0
    # Uma única caixa não é trajetória: ela não recebe evidência de continuidade.
    continuity = exp(-mean_step) if steps else 0.0
    return LocalTrajectory(path, path_score, mean_step, continuity)
