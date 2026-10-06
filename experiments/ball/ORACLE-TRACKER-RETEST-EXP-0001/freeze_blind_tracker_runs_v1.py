"""Validate and hash the completed label-blind tracker inference outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
FREEZE = EXP / "blind_tracker_input_freeze_v1.json"
RUNS = {
    "center_baseline": ("center", "baseline"),
    "center_oversampled": ("center", "oversampled"),
    "full_baseline": ("full", "baseline"),
    "full_oversampled": ("full", "oversampled"),
}
CHECKPOINTS = {
    "baseline": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/baseline/weights/last.pt",
    "oversampled": ROOT / "runs/detect/experiments/ball/BALL-HARD-POSITIVE-EXP-0001/train/oversampled/weights/last.pt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not freeze.get("prediction_only") or freeze.get("labels_included"):
        raise ValueError("Blind input freeze is invalid.")
    records = {}
    for run_name, (window, detector) in RUNS.items():
        input_path = EXP / f"blind_tracker_inputs_{window}_v1.csv"
        run_dir = EXP / run_name
        output_path = run_dir / "frame_identity_results.csv"
        trajectory_path = run_dir / "trajectory_identity_summary.csv"
        metrics_path = run_dir / "metrics.json"
        expected_ids = [row["sample_id"] for row in read(input_path)]
        outputs = read(output_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if [row["sample_id"] for row in outputs] != expected_ids:
            raise ValueError(f"Output sample IDs/order differ from frozen input: {run_name}")
        if not metrics.get("prediction_only") or metrics.get("target_frames") != 0:
            raise ValueError(f"Run was not blind prediction-only: {run_name}")
        if metrics.get("confidence") != 0.01 or metrics.get("max_candidates") != 10 or metrics.get("verifier_checkpoint") is not None:
            raise ValueError(f"Tracker configuration differs from frozen protocol: {run_name}")
        for row in outputs:
            if not row.get("candidate_boxes_json"):
                raise ValueError(f"Missing candidate list for {row['sample_id']} in {run_name}")
        records[run_name] = {
            "window": window,
            "detector": detector,
            "input_path": input_path.name,
            "input_sha256": sha256(input_path),
            "frame_results_path": str(output_path.relative_to(ROOT)),
            "frame_results_rows": len(outputs),
            "frame_results_sha256": sha256(output_path),
            "trajectory_summary_sha256": sha256(trajectory_path),
            "metrics_sha256": sha256(metrics_path),
        }
    manifest = {
        "experiment_id": "ORACLE-TRACKER-RETEST-EXP-0001",
        "prediction_only": True,
        "scope": "TRAIN-only; no VALIDATION/TEST; no V4 promotion or change",
        "input_freeze_sha256": sha256(FREEZE),
        "detector_checkpoints": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in CHECKPOINTS.items()},
        "tracker_configuration": {"confidence": 0.01, "max_candidates": 10, "imgsz": 960, "device": "cpu", "verifier": None},
        "runs": records,
    }
    output = EXP / "blind_tracker_run_freeze_v1.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALIDATED", "runs": len(records), "freeze": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
