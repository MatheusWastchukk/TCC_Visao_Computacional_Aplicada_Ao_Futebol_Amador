"""Mede movimento local na região da goleira como evidência experimental."""
from __future__ import annotations

import cv2
import numpy as np


def stabilized_difference(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Compensa translação/rotação global aproximada antes da diferença."""
    before = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(before, maxCorners=600, qualityLevel=0.01, minDistance=8)
    if points is not None:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None)
        if next_points is not None and status is not None:
            source, target = points[status.ravel() == 1], next_points[status.ravel() == 1]
            if len(source) >= 8:
                matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
                if matrix is not None:
                    before = cv2.warpAffine(before, matrix, (after.shape[1], after.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return cv2.absdiff(before, after)


def net_motion_metrics(difference: np.ndarray, goal_box: tuple[float, float, float, float], margin: float = 0.35) -> tuple[float, float, float]:
    """Retorna movimento interno, anel local e razão interno/anel da goleira."""
    x1, y1, x2, y2 = (int(round(value)) for value in goal_box)
    x1, x2 = max(0, x1), min(difference.shape[1], x2)
    y1, y2 = max(0, y1), min(difference.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0
    width, height = x2 - x1, y2 - y1
    outer_x1, outer_x2 = max(0, int(x1 - width * margin)), min(difference.shape[1], int(x2 + width * margin))
    outer_y1, outer_y2 = max(0, int(y1 - height * margin)), min(difference.shape[0], int(y2 + height * margin))
    inner = difference[y1:y2, x1:x2]
    ring = difference[outer_y1:outer_y2, outer_x1:outer_x2].copy()
    ring[y1 - outer_y1:y2 - outer_y1, x1 - outer_x1:x2 - outer_x1] = 0
    ring_mask = np.ones(ring.shape, dtype=bool)
    ring_mask[y1 - outer_y1:y2 - outer_y1, x1 - outer_x1:x2 - outer_x1] = False
    inside_mean = float(np.mean(inner)) if inner.size else 0.0
    ring_mean = float(np.mean(ring[ring_mask])) if ring_mask.any() else 0.0
    return inside_mean, ring_mean, inside_mean / max(ring_mean, 1e-6)
