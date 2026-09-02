"""Estimativa robusta do movimento global da câmera por Lucas-Kanade."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotion:
    """Deslocamento global estimado entre dois frames consecutivos."""

    dx: float
    dy: float
    valid_points: int
    is_valid: bool


def estimate_camera_motion(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    *,
    max_corners: int = 250,
    quality_level: float = 0.01,
    min_distance: float = 8.0,
    min_points: int = 12,
    max_motion: float = 300.0,
) -> CameraMotion:
    """Estima translação global com mediana e rejeição robusta de outliers.

    O deslocamento segue a convenção ``frame_anterior -> frame_atual``. A
    estimativa é marcada como inválida quando há poucos pontos, quando não há
    fluxo rastreável ou quando a translação excede o limite configurado.
    """
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    points_previous = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
    )
    if points_previous is None:
        return CameraMotion(0.0, 0.0, 0, False)
    points_current, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points_previous,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if points_current is None or status is None:
        return CameraMotion(0.0, 0.0, 0, False)
    vectors = (points_current.reshape(-1, 2) - points_previous.reshape(-1, 2))[status.reshape(-1) == 1]
    if len(vectors) < min_points:
        return CameraMotion(0.0, 0.0, int(len(vectors)), False)
    median = np.median(vectors, axis=0)
    residuals = np.linalg.norm(vectors - median, axis=1)
    median_residual = float(np.median(residuals))
    robust_limit = max(2.0, 3.0 * median_residual)
    inliers = vectors[residuals <= robust_limit]
    if len(inliers) < min_points:
        return CameraMotion(0.0, 0.0, int(len(inliers)), False)
    dx, dy = (float(value) for value in np.median(inliers, axis=0))
    if float(np.hypot(dx, dy)) > max_motion:
        return CameraMotion(0.0, 0.0, int(len(inliers)), False)
    return CameraMotion(dx, dy, int(len(inliers)), True)
