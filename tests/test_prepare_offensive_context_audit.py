from collections import defaultdict

from app.prepare_offensive_context_audit import select_false_candidates


def candidate(match: str, rank: int, start: float, end: float) -> dict[str, str]:
    return {"match_id": match, "temporal_rank": str(rank), "delivery_start_seconds": str(start), "delivery_end_seconds": str(end)}


def test_selection_excludes_windows_that_cover_catalog_goals_and_alternates_matches() -> None:
    rows = [candidate("match_a", 1, 0, 60), candidate("match_a", 2, 100, 160), candidate("match_b", 1, 0, 60)]
    selected = select_false_candidates(rows, defaultdict(list, {"match_a": [30.0], "match_b": []}), 2)
    assert [(row["match_id"], row["temporal_rank"]) for row in selected] == [("match_a", "2"), ("match_b", "1")]
