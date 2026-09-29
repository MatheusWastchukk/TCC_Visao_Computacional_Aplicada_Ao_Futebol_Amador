import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "context_gate", Path("experiments/ball/evaluate_offensive_context_gate_lomo_v1.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_merged_seconds_combines_overlapping_windows() -> None:
    assert MODULE.merged_seconds([(0.0, 10.0), (5.0, 14.0), (20.0, 25.0)]) == 19.0
