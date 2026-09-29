from pathlib import Path

from app.annotate_rank_extremes_audit import merge_existing


def test_audit_annotation_resumes_primary_factor(tmp_path: Path) -> None:
    output = tmp_path / "annotations.csv"
    output.write_text(
        "audit_id,primary_factor,annotation_status\ncase_01,SINAL_LIMPO,completed\n",
        encoding="utf-8",
    )
    rows, fields = merge_existing(
        [{"audit_id": "case_01", "clip_filename": "case.mp4", "match_id": "match_01", "group": "TEST"}],
        ["audit_id", "clip_filename", "match_id", "group"],
        output,
    )
    assert "primary_factor" in fields
    assert rows[0]["primary_factor"] == "SINAL_LIMPO"
    assert rows[0]["annotation_status"] == "completed"
