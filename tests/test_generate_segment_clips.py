from app.generate_segment_clips import clip_bounds


def test_clip_bounds_limits_interval_to_video() -> None:
    assert clip_bounds(-4.0, 12.0, 10.0) == (0.0, 10.0)
