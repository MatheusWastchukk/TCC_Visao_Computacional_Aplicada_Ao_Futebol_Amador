"""Create label-free sequence CSVs for frozen TRAIN tracker inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
ANNOTATIONS = EXP / "frame_annotations_v1.csv"
EVENTS = EXP / "event_manifest_v1.csv"
SAFE_FIELDS = ["sequence_id", "sequence_frame", "sample_id", "image_path", "match_id", "event_role", "pair_id"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAFE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    annotations, events = read(ANNOTATIONS), read(EVENTS)
    event_by_sequence = {row["sequence_id"]: row for row in events}
    safe_rows = []
    for row in annotations:
        # Explicit whitelist: no visibility, boxes, goal geometry, or annotation status enters inference.
        safe_rows.append({field: row.get(field, "") for field in SAFE_FIELDS})
    safe_rows.sort(key=lambda row: (row["sequence_id"], int(row["sequence_frame"])))
    if len(safe_rows) != len(events) * 41 or {r["sequence_id"] for r in safe_rows} != set(event_by_sequence):
        raise ValueError("Sequence rows do not match the frozen 35 x 41 event manifest.")
    for row in safe_rows:
        if not (Path(row["image_path"]) if Path(row["image_path"]).is_absolute() else EXP.parents[2] / row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    outputs = {}
    for name, subset in (("full", safe_rows), ("center", [r for r in safe_rows if 16 <= int(r["sequence_frame"]) <= 26])):
        path = EXP / f"blind_tracker_inputs_{name}_v1.csv"
        write(path, subset)
        outputs[name] = {"path": path.name, "rows": len(subset), "sha256": digest(path)}
    freeze = {
        "experiment_id": "ORACLE-TRACKER-RETEST-EXP-0001",
        "prediction_only": True,
        "labels_included": False,
        "scope": "TRAIN-only engineering inference; no TEST/VALIDATION and no V4 change",
        "event_manifest_sha256": digest(EVENTS),
        "input_source_sha256_at_preparation": digest(ANNOTATIONS),
        "input_columns": SAFE_FIELDS,
        "outputs": outputs,
        "tracker_configuration": {"confidence": 0.01, "max_candidates": 10, "imgsz": 960, "device": "cpu", "verifier": None},
    }
    (EXP / "blind_tracker_input_freeze_v1.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
