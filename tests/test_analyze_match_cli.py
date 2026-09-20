import csv
import json
from pathlib import Path

from app.analyze_match import main


def test_analyze_match_accepts_arguments_without_subprocess(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    with candidates.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "event_id", "match_id", "video_filename", "event_timestamp_seconds", "rank_score",
                "fusion_score_goal", "delivery_start_seconds", "delivery_end_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_id": "event_001", "match_id": "match_demo", "video_filename": "demo.mp4",
                "event_timestamp_seconds": "20", "rank_score": "0.8", "fusion_score_goal": "0.7",
                "delivery_start_seconds": "0", "delivery_end_seconds": "60",
            }
        )
    output = tmp_path / "output"

    assert main(["--candidate-scores", str(candidates), "--mode", "alta_cobertura", "--output", str(output)]) == 0

    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert result["summary"]["segments_after_merge"] == 1
