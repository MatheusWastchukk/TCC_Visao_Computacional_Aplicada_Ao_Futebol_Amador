"""Atributos cinemáticos compatíveis com movimento ofensivo da bola.

Não detecta nem confirma um chute. Produz evidência numérica para uma hipótese
de ranking, a ser avaliada isoladamente em desenvolvimento.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from src.events.local_trajectory_v1 import LocalTrajectory


@dataclass(frozen=True)
class ShotMotionEvidence:
    """Resumo geométrico de uma trajetória em relação ao centro da goleira."""

    path_observations: int
    mean_speed: float
    max_speed: float
    mean_toward_goal_cosine: float
    approach_fraction: float
    normalized_distance_delta: float


def shot_motion_evidence(trajectory: LocalTrajectory) -> ShotMotionEvidence:
    """Mede velocidade e direção sem transformar o sinal em decisão de chute."""
    path = trajectory.observations
    transitions: list[tuple[float, float]] = []
    for first, second in zip(path, path[1:]):
        elapsed = second.timestamp_seconds - first.timestamp_seconds
        movement_x, movement_y = second.x - first.x, second.y - first.y
        goal_x, goal_y = first.goal_x - first.x, first.goal_y - first.y
        movement_norm, goal_norm = hypot(movement_x, movement_y), hypot(goal_x, goal_y)
        if elapsed <= 0 or movement_norm == 0 or goal_norm == 0:
            continue
        speed = movement_norm / max(first.reference_scale, second.reference_scale, 1.0) / elapsed
        cosine = (movement_x * goal_x + movement_y * goal_y) / (movement_norm * goal_norm)
        transitions.append((speed, max(-1.0, min(1.0, cosine))))
    if not transitions:
        return ShotMotionEvidence(len(path), 0.0, 0.0, 0.0, 0.0, 0.0)
    first, last = path[0], path[-1]
    first_distance = hypot(first.goal_x - first.x, first.goal_y - first.y) / max(first.reference_scale, 1.0)
    last_distance = hypot(last.goal_x - last.x, last.goal_y - last.y) / max(last.reference_scale, 1.0)
    speeds, cosines = zip(*transitions)
    return ShotMotionEvidence(
        path_observations=len(path),
        mean_speed=sum(speeds) / len(speeds),
        max_speed=max(speeds),
        mean_toward_goal_cosine=sum(cosines) / len(cosines),
        approach_fraction=sum(cosine >= 0.5 for cosine in cosines) / len(cosines),
        normalized_distance_delta=first_distance - last_distance,
    )
