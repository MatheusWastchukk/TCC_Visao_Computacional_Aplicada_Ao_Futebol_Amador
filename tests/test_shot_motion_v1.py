from src.events.local_trajectory_v1 import BallObservation, LocalTrajectory
from src.events.shot_motion_v1 import shot_motion_evidence


def trajectory(points: list[tuple[float, float]]) -> LocalTrajectory:
    observations = tuple(
        BallObservation(float(index), x, y, 0.9, 100.0, 100.0, 0.0)
        for index, (x, y) in enumerate(points)
    )
    return LocalTrajectory(observations, 0.0, 0.0, 0.0)


def test_shot_motion_detects_motion_toward_goal_without_deciding_shot() -> None:
    evidence = shot_motion_evidence(trajectory([(0.0, 0.0), (40.0, 0.0), (80.0, 0.0)]))
    assert evidence.mean_speed == 0.4
    assert evidence.approach_fraction == 1.0
    assert evidence.normalized_distance_delta == 0.8


def test_shot_motion_has_no_evidence_for_single_observation() -> None:
    evidence = shot_motion_evidence(trajectory([(0.0, 0.0)]))
    assert evidence.path_observations == 1
    assert evidence.max_speed == 0.0
