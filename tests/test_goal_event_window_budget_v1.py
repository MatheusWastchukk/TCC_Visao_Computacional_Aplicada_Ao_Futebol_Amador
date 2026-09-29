import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("window_budget", Path("experiments/ball/evaluate_goal_event_window_budget_v1.py"))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_merge_combines_overlapping_intervals() -> None:
    assert MODULE.merge([(0.0, 10.0), (5.0, 15.0), (20.0, 25.0)]) == [(0.0, 15.0), (20.0, 25.0)]


def test_parse_window() -> None:
    assert MODULE.parse_window("5:25") == (5.0, 25.0)
