from __future__ import annotations

import numpy as np

from experiments.goal_detection_pilot.scripts.run_baseline import (
    calculate_metrics,
    threshold_mask,
)


def test_threshold_mask_detects_bright_pixels() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:10, 5:10] = (255, 255, 255)

    mask = threshold_mask(image, {"color_space": "gray", "method": "fixed", "min_value": 190})

    assert int(mask[7, 7]) == 255
    assert int(mask[0, 0]) == 0


def test_calculate_metrics_treats_secondary_goal_as_negative_for_main_goal() -> None:
    rows = [
        {"label": "main_goal", "expected": True, "predicted": True},
        {"label": "not_goal", "expected": False, "predicted": False},
        {"label": "secondary_goal", "expected": False, "predicted": True},
    ]

    metrics = calculate_metrics(rows)

    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.secondary_rejection_rate == 0.0
