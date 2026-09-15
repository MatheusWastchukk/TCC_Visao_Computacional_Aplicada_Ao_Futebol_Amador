from pathlib import Path

from app.annotate_segment_clips import merge_existing, output_fields, read_rows, write_rows


def test_annotation_output_preserves_manifest_and_resumes(tmp_path: Path) -> None:
    manifest = tmp_path / "clips_manifest.csv"
    manifest.write_text(
        "segment_id,clip_filename,label\nsegment_001,001.mp4,POSSIBLE_GOAL\n",
        encoding="utf-8",
    )
    source_rows, source_fields = read_rows(manifest)
    fields = output_fields(source_fields)
    output = tmp_path / "clip_annotations.csv"
    rows = merge_existing(source_rows, fields, output)
    rows[0]["human_label"] = "GOAL"
    rows[0]["annotation_status"] = "completed"
    write_rows(output, fields, rows)
    resumed = merge_existing(source_rows, fields, output)
    assert resumed[0]["human_label"] == "GOAL"
    assert resumed[0]["annotation_status"] == "completed"
