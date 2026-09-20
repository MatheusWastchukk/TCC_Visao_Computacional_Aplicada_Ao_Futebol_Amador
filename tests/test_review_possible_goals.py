from pathlib import Path

from app.review_storage import load_annotation_rows, safe_run_directory


def test_safe_run_directory_stays_below_results_root(tmp_path: Path) -> None:
    destination = safe_run_directory(
        tmp_path / "results" / "app_runs", "match_04", "alta_cobertura"
    )

    assert destination == (tmp_path / "results" / "app_runs" / "match_04_alta_cobertura").resolve()


def test_review_annotations_resume_without_changing_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "clips_manifest.csv"
    manifest.write_text("segment_id,clip_filename\nsegment_001,001.mp4\n", encoding="utf-8")
    output = tmp_path / "clip_annotations.csv"
    output.write_text(
        "segment_id,clip_filename,human_label,annotation_status,notes\n"
        "segment_001,001.mp4,NO_GOAL,completed,sem finalização\n",
        encoding="utf-8",
    )

    rows, fields = load_annotation_rows(manifest, output)

    assert fields[-3:] == ["human_label", "annotation_status", "notes"]
    assert rows[0]["human_label"] == "NO_GOAL"
    assert rows[0]["annotation_status"] == "completed"
