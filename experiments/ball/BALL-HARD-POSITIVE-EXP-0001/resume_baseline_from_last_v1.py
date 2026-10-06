"""Resume the frozen BALL hard-positive baseline from its saved optimizer state."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "ball" / "BALL-HARD-POSITIVE-EXP-0001"
LAST = ROOT / "runs" / "detect" / "experiments" / "ball" / "BALL-HARD-POSITIVE-EXP-0001" / "train" / "baseline" / "weights" / "last.pt"
RESULTS = LAST.parents[1] / "results.csv"
META = EXP / "train" / "baseline" / "metrics.json"
CONFIG = ROOT / ".tmp" / "ultralytics_config_v2"


def main() -> int:
    os.environ["YOLO_CONFIG_DIR"] = str(CONFIG.resolve())
    if META.exists():
        raise RuntimeError(f"Refusing to resume a baseline already marked complete: {META}")
    if not LAST.is_file() or LAST.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Missing or implausible resume checkpoint: {LAST}")

    from ultralytics import YOLO

    model = YOLO(str(LAST))
    ckpt = model.ckpt or {}
    args = ckpt.get("train_args") or {}
    expected = {
        "data": r"experiments\ball\BALL-HARD-POSITIVE-EXP-0001\dataset_baseline.yaml",
        "epochs": 30,
        "imgsz": 960,
        "batch": 8,
        "device": "cpu",
        "workers": 0,
        "val": False,
        "amp": False,
    }
    mismatches = {key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Checkpoint settings differ from the frozen protocol: {mismatches}")
    if ckpt.get("optimizer") is None or int(ckpt.get("epoch", -1)) < 0:
        raise RuntimeError("Checkpoint lacks resumable epoch and optimizer state.")
    start_epoch = int(ckpt["epoch"]) + 1

    started = time.perf_counter()
    results = model.train(
        resume=True,
        epochs=30,
        imgsz=960,
        batch=8,
        device="cpu",
        workers=0,
        val=False,
        amp=False,
        verbose=False,
    )
    elapsed = round(time.perf_counter() - started, 3)

    if not RESULTS.is_file() or not LAST.is_file():
        raise RuntimeError("Resumed training returned without the expected results.csv and last.pt.")
    with RESULTS.open("r", encoding="utf-8-sig", newline="") as handle:
        last_row = next(csv.reader(handle), None)
        for row in csv.reader(handle):
            if row:
                last_row = row
    if not last_row or int(float(last_row[0])) != 30:
        raise RuntimeError(f"Expected fixed epoch number 30 after resume; final results row is {last_row}.")
    if META.exists():
        raise RuntimeError(f"Refusing to overwrite completion metadata created concurrently: {META}")

    raw_metrics = dict(getattr(results, "results_dict", {}) or {})
    summary = {
        "experiment_id": "baseline",
        "model": "yolo11n.pt",
        "data": expected["data"],
        "epochs": 30,
        "imgsz": 960,
        "batch": 8,
        "device": "cpu",
        "metrics_scope": "post-training training-fold diagnostics only; not the frozen holdout comparison or a promotion metric",
        "metrics": {key: float(value) for key, value in raw_metrics.items() if isinstance(value, (int, float))},
        "resume_checkpoint": str(LAST.relative_to(ROOT)),
        "resume_from_checkpoint_epoch": int(ckpt["epoch"]),
        "resume_execution_seconds": elapsed,
        "status": "completed",
    }
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
