"""Apply the preregistered detector gate to one frozen holdout comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
DEFAULT_EVAL = EXP / "holdout_evaluation_v1"
EXPECTED_MATCHES = ["match_10", "match_16", "match_23", "match_27", "match_36"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    protocol_path = evaluation_dir / "protocol_v1.json"
    summary_path = evaluation_dir / "holdout_summary_v1.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation_artifacts = protocol.get("evaluation_artifacts") or {}
    for filename in ("holdout_frame_results_v1.csv", "holdout_summary_v1.csv"):
        artifact = evaluation_artifacts.get(filename) or {}
        expected_path = (evaluation_dir / filename).resolve()
        if Path(artifact.get("path", "")).resolve() != expected_path:
            raise ValueError(f"Evaluation artifact path mismatch for {filename}: {artifact.get('path')}")
        if not expected_path.is_file() or artifact.get("sha256") != sha256(expected_path):
            raise ValueError(f"Evaluation artifact hash mismatch for {filename}.")
    expected_protocol = {
        "holdout_rows": 150,
        "confidence": 0.01,
        "top_n": 10,
        "iou_threshold": 0.50,
        "imgsz": 960,
        "device": "CPU",
        "positive_area_threshold_px2": 400.0,
    }
    mismatches = {
        key: (protocol.get(key), expected)
        for key, expected in expected_protocol.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Holdout evaluation protocol differs from the frozen detector gate: {mismatches}")
    if protocol.get("scope") != "five untouched TRAIN matches only; not VALIDATION or TEST and not an external generalization estimate":
        raise ValueError("Evaluation scope is not the expected frozen TRAIN holdout.")
    if protocol.get("holdout_matches") != EXPECTED_MATCHES:
        raise ValueError(f"Holdout matches differ from the frozen split: {protocol.get('holdout_matches')}")
    annotation_source = protocol.get("annotation_source") or {}
    expected_annotation_path = (EXP / "holdout_train_annotations_v1.csv").resolve()
    if Path(annotation_source.get("path", "")).resolve() != expected_annotation_path:
        raise ValueError(f"Holdout annotation file differs from the frozen source: {annotation_source.get('path')}")
    if not expected_annotation_path.is_file() or annotation_source.get("sha256") != sha256(expected_annotation_path):
        raise ValueError("Holdout annotation hash differs from the evaluated frozen source.")
    if protocol.get("positive_area_group_counts") != {"lt_400_px2": 9, "ge_400_px2": 95}:
        raise ValueError(f"Positive holdout area-stratum counts differ from the addendum: {protocol.get('positive_area_group_counts')}")
    if protocol.get("candidate_recall") != "fraction of VISIBLE/PARTIAL frames with any of top-N predictions at IoU >= threshold":
        raise ValueError("Candidate-recall definition differs from the frozen gate.")
    addendum = protocol.get("analysis_addendum") or {}
    addendum_path = (ROOT / "experiments" / "ball" / "BALL-HARD-POSITIVE-EXP-0001" / "protocol_addendum_area_stratification_v1.json").resolve()
    addendum_recorded_path = Path(addendum.get("path", "")).resolve()
    if addendum_recorded_path != addendum_path or not addendum_path.is_file() or addendum.get("sha256") != sha256(addendum_path):
        raise ValueError("The descriptive area-stratification addendum is missing or differs from the reviewed protocol.")

    expected_checkpoints = {
        "baseline": ROOT / "runs" / "detect" / "experiments" / "ball" / "BALL-HARD-POSITIVE-EXP-0001" / "train" / "baseline" / "weights" / "last.pt",
        "oversampled": ROOT / "runs" / "detect" / "experiments" / "ball" / "BALL-HARD-POSITIVE-EXP-0001" / "train" / "oversampled" / "weights" / "last.pt",
    }
    recorded_checkpoints = protocol.get("checkpoints") or {}
    if set(recorded_checkpoints) != set(expected_checkpoints):
        raise ValueError(f"Expected exactly the frozen baseline and oversampled checkpoints; found {sorted(recorded_checkpoints)}")
    for model_id, expected_path in expected_checkpoints.items():
        recorded = recorded_checkpoints[model_id]
        if Path(recorded.get("path", "")).resolve() != expected_path.resolve():
            raise ValueError(f"Checkpoint path mismatch for {model_id}: {recorded.get('path')}")
        if not expected_path.is_file() or recorded.get("sha256") != sha256(expected_path):
            raise ValueError(f"Checkpoint hash mismatch for {model_id}: {expected_path}")

    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    overall = {
        row["model_id"]: row
        for row in rows
        if row.get("group_type") == "all_visible_partial" and row.get("group_value") == "all"
    }
    if set(overall) != {"baseline", "oversampled"}:
        raise ValueError(f"Expected one overall positive summary per model; found {sorted(overall)}")

    parsed: dict[str, dict[str, float | int]] = {}
    for model_id, row in overall.items():
        positive_frames = int(row["positive_frames"])
        hits = int(row["hits_top_n_iou50"])
        recall = float(row["candidate_recall"])
        if positive_frames != 104 or not 0 <= hits <= positive_frames:
            raise ValueError(f"Unexpected positive holdout denominator/count for {model_id}: {row}")
        if not math.isclose(recall, hits / positive_frames, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Candidate recall does not match hits/frames for {model_id}: {row}")
        parsed[model_id] = {"positive_frames": positive_frames, "hits": hits, "candidate_recall": recall}
    if parsed["baseline"]["positive_frames"] != parsed["oversampled"]["positive_frames"]:
        raise ValueError("Models were not compared on the same positive holdout rows.")

    baseline_recall = float(parsed["baseline"]["candidate_recall"])
    oversampled_recall = float(parsed["oversampled"]["candidate_recall"])
    threshold = 0.50
    gate_passed = oversampled_recall >= threshold and oversampled_recall > baseline_recall
    result = {
        "experiment_id": "BALL-HARD-POSITIVE-EXP-0001",
        "scope": "TRAIN-only engineering gate; no VALIDATION/TEST or generalization claim",
        "evaluation_protocol": str(protocol_path.relative_to(ROOT)),
        "holdout_summary": str(summary_path.relative_to(ROOT)),
        "gate": {
            "criterion": "oversampled positive candidate recall >= 0.50 and greater than baseline on identical holdout rows",
            "minimum_oversampled_recall": threshold,
            "baseline": parsed["baseline"],
            "oversampled": parsed["oversampled"],
            "passed": gate_passed,
            "decision": "REOPEN_TRACKER" if gate_passed else "NO_GO_TRACKER",
        },
        "weight_selection": "No checkpoint selection or detector promotion is permitted from this comparison.",
        "next_step": "Prepare the conditional same-Viterbi TRAIN retest only if the gate passes; otherwise keep tracking closed.",
    }
    output = args.output.resolve() if args.output else evaluation_dir / "tracker_gate_decision_v1.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
