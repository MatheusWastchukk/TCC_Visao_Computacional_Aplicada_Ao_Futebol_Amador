from src.events.local_trajectory_v1 import BallObservation, best_local_trajectory, observations_from_frame_rows


def observation(timestamp: float, x: float, confidence: float) -> BallObservation:
    return BallObservation(timestamp, x, 100.0, confidence, 100.0)


def test_best_local_trajectory_prefers_continuity_over_distant_detection() -> None:
    result = best_local_trajectory([
        observation(0.0, 0.0, 0.9),
        observation(0.5, 8.0, 0.9),
        observation(0.5, 300.0, 0.99),
        observation(1.0, 16.0, 0.9),
    ])
    assert [item.x for item in result.observations] == [0.0, 8.0, 16.0]
    assert result.continuity > 0.9


def test_best_local_trajectory_returns_empty_summary_without_observations() -> None:
    result = best_local_trajectory([])
    assert result.observations == ()
    assert result.path_score == 0.0


def test_single_observation_has_no_continuity_evidence() -> None:
    result = best_local_trajectory([observation(0.0, 10.0, 0.8)])
    assert result.continuity == 0.0


def test_observations_from_frame_rows_uses_existing_miner_columns() -> None:
    rows = [{
        "timestamp_seconds": "4.5",
        "ball_box_json": "[10, 20, 14, 26]",
        "goal_box_json": "[0, 0, 300, 400]",
        "ball_confidence": "0.8",
    }]
    result = observations_from_frame_rows(rows)
    assert result[0] == BallObservation(4.5, 12.0, 23.0, 0.8, 500.0, 150.0, 200.0)
