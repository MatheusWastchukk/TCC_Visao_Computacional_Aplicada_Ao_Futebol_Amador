"""Validate complete human annotations and freeze full/central tracker inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parent
ANNOTATIONS = EXP / "frame_annotations_v1.csv"
EVENTS = EXP / "event_manifest_v1.csv"
PROTOCOL = EXP / "protocol_v1.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No scoring rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--output-dir", type=Path, default=EXP)
    parser.add_argument(
        "--ball-identity-only",
        action="store_true",
        help="Freeze BALL candidate/identity labels while leaving goal geometry explicitly unscored.",
    )
    args = parser.parse_args()
    rows = read_csv(args.annotations)
    events = read_csv(args.events)
    by_sequence: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_sequence.setdefault(row["sequence_id"], []).append(row)
    expected_sequences = {row["sequence_id"] for row in events}
    errors: list[str] = []
    if len(events) != 35 or len(rows) != 35 * 41 or set(by_sequence) != expected_sequences:
        errors.append(f"Expected 35 events/sequences and 1,435 frames; got {len(events)}, {len(by_sequence)}, {len(rows)}")
    for event in events:
        sequence = sorted(by_sequence.get(event["sequence_id"], []), key=lambda row: int(row["sequence_frame"]))
        if len(sequence) != 41 or [int(row["sequence_frame"]) for row in sequence] != list(range(1, 42)):
            errors.append(f"Bad frame-index sequence for {event['sequence_id']}")
            continue
        if any(row["annotation_status"] != "completed" for row in sequence):
            errors.append(f"Incomplete BALL labels for {event['sequence_id']}")
        for row in sequence:
            visibility = row.get("ball_visibility", "")
            if visibility not in {"VISIBLE", "PARTIAL", "NOT_VISIBLE", "UNCERTAIN"}:
                errors.append(f"Missing/invalid visibility on {row['sample_id']}: {visibility!r}")
            if visibility in {"VISIBLE", "PARTIAL"} and not row.get("ball_of_play_box_json"):
                errors.append(f"Visible ball has no box on {row['sample_id']}")
            if not (Path(row["image_path"]) if Path(row["image_path"]).is_absolute() else Path(__file__).resolve().parents[3] / row["image_path"]).is_file():
                errors.append(f"Missing frame image {row['image_path']}")
        center = sequence[20]
        if not args.ball_identity_only and center.get("goal_region_status") not in {"VALID", "NOT_VISIBLE", "UNCERTAIN"}:
            errors.append(f"Missing goal-region decision at event center for {event['sequence_id']}")
    if errors:
        raise ValueError("Annotation freeze rejected:\n- " + "\n- ".join(errors[:30]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_rows = sorted(rows, key=lambda row: (row["sequence_id"], int(row["sequence_frame"])))
    center_rows = [row for row in full_rows if 16 <= int(row["sequence_frame"]) <= 26]
    if len(center_rows) != 35 * 11:
        raise AssertionError(f"Expected 385 central frames; got {len(center_rows)}")
    full_path = args.output_dir / "tracking_scoring_frames_full_v1.csv"
    center_path = args.output_dir / "tracking_scoring_frames_center_v1.csv"
    write_csv(full_path, full_rows)
    write_csv(center_path, center_rows)
    frozen = {
        "experiment_id": "ORACLE-TRACKER-RETEST-EXP-0001",
        "scope": "TRAIN only; TEST remains closed",
        "scoring_scope": "BALL identity/candidate metrics only; goal geometry is not evaluated" if args.ball_identity_only else "BALL identity plus completed GOAL geometry decisions",
        "ball_identity_only": args.ball_identity_only,
        "annotations_path": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "event_manifest_path": str(args.events.resolve()),
        "event_manifest_sha256": sha256(args.events),
        "full_input": {"path": str(full_path.resolve()), "rows": len(full_rows), "sha256": sha256(full_path)},
        "center_input": {"path": str(center_path.resolve()), "frames_1_based": [16, 26], "rows": len(center_rows), "sha256": sha256(center_path)},
        "sequences": len(events),
        "visibility_denominators": {
            status: sum(row["ball_visibility"] == status for row in rows)
            for status in ("VISIBLE", "PARTIAL", "NOT_VISIBLE", "UNCERTAIN")
        },
        "goal_region_status_at_centers": {
            status: sum(row["is_event_frame"] == "True" and row["goal_region_status"] == status for row in rows)
            for status in ("VALID", "NOT_VISIBLE", "UNCERTAIN")
        },
        "goal_geometry_missing_decisions_at_centers": sum(
            row["is_event_frame"] == "True" and row.get("goal_region_status", "") not in {"VALID", "NOT_VISIBLE", "UNCERTAIN"}
            for row in rows
        ),
        "tracker_configuration": {"confidence": 0.01, "max_candidates": 10, "imgsz": 960, "device": "cpu", "verifier": None},
    }
    (args.output_dir / "annotation_freeze_v1.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "full_rows": len(full_rows), "center_rows": len(center_rows), "freeze": str(args.output_dir / 'annotation_freeze_v1.json')}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
