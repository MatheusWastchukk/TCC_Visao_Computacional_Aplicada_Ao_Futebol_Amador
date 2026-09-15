from src.events.operational_modes_v4 import analyze_match


def candidate(index: int, start: float, fusion: float, temporal: float) -> dict[str, str]:
    return {
        "event_id": f"event_{index:03d}", "match_id": "match_demo", "video_filename": "demo.mp4",
        "event_timestamp_seconds": str(start + 20.0), "rank_score": str(temporal),
        "fusion_score_goal": str(fusion), "delivery_start_seconds": str(start), "delivery_end_seconds": str(start + 60.0),
    }


def test_hybrid_selects_fusion_core_and_temporal_complements() -> None:
    rows = [candidate(index, index * 100.0, 100.0 - index, float(index)) for index in range(60)]
    result = analyze_match(rows, "cobertura_reduzida")
    selected = result["selected_candidates"]
    assert len(selected) == 45
    ids = {row["event_id"] for row in selected}
    assert "event_000" in ids
    assert "event_059" in ids


def test_merge_unites_overlapping_delivery_intervals() -> None:
    rows = [candidate(1, 0.0, 0.9, 0.8), candidate(2, 30.0, 0.8, 0.7), candidate(3, 200.0, 0.7, 0.6)]
    result = analyze_match(rows, "triagem_rapida")
    segments = result["segments"]
    assert len(segments) == 2
    assert segments[0]["duration_seconds"] == 90.0
    assert result["summary"]["unique_review_seconds"] == 150.0
