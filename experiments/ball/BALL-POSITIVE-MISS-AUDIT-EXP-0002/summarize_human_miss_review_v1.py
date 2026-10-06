"""Summarize the completed human review of the frozen 127 BALL misses."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parent
INPUT = EXP / "positive_miss_mechanism_v2_annotated.csv"
OUTPUT_JSON = EXP / "human_visual_review_summary_v1.json"
OUTPUT_MD = EXP / "human_visual_review_summary_v1.md"
TAGS = (
    "SMALL_BALL",
    "BLUR",
    "OCCLUSION_PLAYER",
    "OCCLUSION_GOALKEEPER",
    "OCCLUSION_NET",
    "LOW_CONTRAST",
    "FRAME_EDGE",
    "BAD_LOCALIZATION",
    "OTHER",
    "UNCERTAIN",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    with INPUT.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 127:
        raise ValueError(f"Expected the frozen set of 127 misses; found {len(rows)}.")
    incomplete = [row["sequence_id"] for row in rows if row["visual_review_status"] != "COMPLETED"]
    if incomplete:
        raise ValueError(f"Human visual review is incomplete on {len(incomplete)} frames.")

    counts = Counter()
    combinations = Counter()
    for row in rows:
        labels = [item for item in (row.get("visual_cause_tags") or "").split(";") if item]
        unknown = sorted(set(labels) - set(TAGS))
        if unknown:
            raise ValueError(f"Unknown visual tags at {row['sequence_id']}: {unknown}")
        combinations[";".join(labels) if labels else "(blank)"] += 1
        counts.update(set(labels))

    summary = {
        "experiment_id": "BALL-POSITIVE-MISS-AUDIT-EXP-0002",
        "scope": "TRAIN-only descriptive summary of the 127 frozen positive miss frames",
        "input_path": str(INPUT.relative_to(Path(__file__).resolve().parents[3])),
        "input_sha256": sha256(INPUT),
        "frames": len(rows),
        "completed_visual_reviews": len(rows) - len(incomplete),
        "frames_with_uncertain_tag": counts["UNCERTAIN"],
        "frames_with_at_least_one_specific_visual_tag": sum(
            bool(set((row.get("visual_cause_tags") or "").split(";")) - {"", "UNCERTAIN"}) for row in rows
        ),
        "tag_counts_overlapping": {tag: counts[tag] for tag in TAGS},
        "tag_combination_counts": dict(sorted(combinations.items())),
        "objective_proposal_status": dict(Counter(row["proposal_status"] for row in rows)),
        "objective_selected_tracker_box_status": dict(Counter(row["selected_tracker_box_status"] for row in rows)),
        "interpretation_limits": [
            "Visual tags overlap and summarize reviewer judgments on this selected miss cohort.",
            "The audit includes misses only, so tag frequency is not a miss-versus-recovery prevalence comparison.",
            "Tags are descriptive observations, not causal estimates or automatic training labels.",
            "Proposal count and selected tracker-box status remain separate objective measurements.",
            "The Codex contact-sheet screening is stored separately and is not merged into these labels.",
            "TRAIN only; no VALIDATION or TEST access.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Completed human visual review of 127 BALL misses",
        "",
        "Scope: TRAIN-only; frozen miss cohort; overlapping descriptive tags, not causal prevalence.",
        "",
        f"- Reviewed: {summary['completed_visual_reviews']}/{summary['frames']}",
        f"- UNCERTAIN: {summary['frames_with_uncertain_tag']}",
        f"- Frames with at least one specific visual tag: {summary['frames_with_at_least_one_specific_visual_tag']}",
        "",
        "| Tag | Frames |",
        "|---|---:|",
    ]
    lines.extend(f"| {tag} | {counts[tag]} |" for tag in TAGS)
    lines.extend([
        "",
        "These counts overlap. The 127-frame set contains misses only and cannot establish that any visual factor is more common among misses than recovered positives. The tag counts must not be treated as causal proof or general population rates.",
        "",
        f"Input SHA-256: `{summary['input_sha256']}`",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "SUMMARIZED", "frames": len(rows), "json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
