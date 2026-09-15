from src.events.temporal_inference_v4 import temporal_groups


def frame(timestamp: float) -> dict[str, str]:
    return {"match_id": "match_demo", "video_filename": "demo.mp4", "timestamp_seconds": str(timestamp), "candidate_score": "0.8", "ball_confidence": "0.4", "goal_confidence": "0.9", "normalized_distance": "0.3", "ball_inside_expanded_goal": "True", "ball_box_json": "[0, 0, 10, 10]"}


def test_temporal_groups_builds_frozen_feature_contract() -> None:
    groups = temporal_groups([frame(1.0), frame(1.5), frame(2.0)])
    assert len(groups) == 1
    assert groups[0]["hit_count"] == 3.0
    assert "bin_3_inside_fraction" in groups[0]
