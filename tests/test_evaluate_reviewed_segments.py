from app.evaluate_reviewed_segments import evaluate


def test_evaluate_associates_confirmed_segment_to_catalog_goal() -> None:
    result = evaluate(
        [{"segment_id": "segment_001", "start_seconds": "10", "end_seconds": "70"}],
        [{"segment_id": "segment_001", "human_label": "GOAL"}],
        [
            {
                "goal_catalog_id": "goal_001",
                "annotation_status": "completed",
                "event_timestamp_seconds": "30",
            }
        ],
    )
    assert result["completed_catalog_goals"] == 1
    assert result["covered_catalog_goals"] == 1
    assert result["human_confirmed_catalog_goals"] == 1
    assert result["reviewed_goal_segments_associated_with_catalog"] == 1
