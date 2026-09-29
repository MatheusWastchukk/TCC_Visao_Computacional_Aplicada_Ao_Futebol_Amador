import numpy as np

from src.events.net_motion_v1 import net_motion_metrics


def test_net_motion_metrics_favors_change_inside_goal_region() -> None:
    difference = np.zeros((100, 100), dtype=np.uint8)
    difference[40:60, 40:60] = 100
    inside, ring, ratio = net_motion_metrics(difference, (40, 40, 60, 60))
    assert inside == 100.0
    assert ring == 0.0
    assert ratio > 1_000_000


def test_net_motion_metrics_rejects_invalid_goal_box() -> None:
    assert net_motion_metrics(np.zeros((10, 10), dtype=np.uint8), (4, 4, 4, 8)) == (0.0, 0.0, 0.0)
