"""Extrai frames do piloto oracle e reaproveita anotacoes manuais exatas existentes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
PAIR_MANIFEST = EXP / "oracle_pair_manifest_v1.csv"
PRIOR_TRACKS = ROOT / "experiments/ball/BALL-EVENT-TRAJECTORY-DATASET-V1/event_trajectory_sequences_v1.csv"
VIDEOS = ROOT / "data/Videos_Edited"
FRAMES_DIR = EXP / "oracle_frames_v1"
DEFAULT_OUTPUT = EXP / "oracle_frame_annotations_v1.csv"
SECONDS_BEFORE = 4.0
SECONDS_AFTER = 4.0
SAMPLE_FPS = 5.0
REUSE_TOLERANCE_SECONDS = 0.05

EXTRA_FIELDS = [
    "goal_region_points_json", "goal_region_status", "goal_volume_points_json", "goal_volume_status",
    "annotation_source", "annotation_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def old_manual_by_match(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("annotation_status") == "completed" and row.get("annotator") == "manual":
            grouped.setdefault(row["match_id"], []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: float(row["timestamp_seconds"]))
    return grouped


def format_time(seconds: float) -> str:
    value = max(0, round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PAIR_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--seconds-before", type=float, default=SECONDS_BEFORE)
    parser.add_argument("--seconds-after", type=float, default=SECONDS_AFTER)
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.seconds_before <= 0 or args.seconds_after <= 0:
        parser.error("sample-fps e duracoes da janela devem ser positivos")

    pairs = read_csv(args.manifest)
    old_rows = read_csv(PRIOR_TRACKS) if PRIOR_TRACKS.is_file() else []
    old_by_match = old_manual_by_match(old_rows)
    previous = {row["sample_id"]: row for row in read_csv(args.output)} if args.output.is_file() else {}
    results: list[dict[str, str]] = []
    generated = reused = 0

    for pair in pairs:
        for role, event_id, timestamp_key, label in (
            ("POSITIVE", f"{pair['pair_id']}_POS", "positive_timestamp_seconds", "CATALOGED_GOAL"),
            ("NEGATIVE", f"{pair['pair_id']}_NEG", "negative_timestamp_seconds", "CONFIRMED_NO_GOAL"),
        ):
            match = pair["match_id"]
            video_name = pair["video_filename"]
            video_path = args.videos_dir / video_name
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Video TRAIN nao encontrado: {video_path}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps <= 0 or frame_count <= 0:
                    raise ValueError(f"Metadados invalidos no video: {video_path}")
                duration = (frame_count - 1) / fps
                center = float(pair[timestamp_key])
                offsets: list[float] = []
                step = 1.0 / args.sample_fps
                count = round((args.seconds_before + args.seconds_after) * args.sample_fps)
                for sample_index in range(count + 1):
                    offsets.append(-args.seconds_before + sample_index * step)
                event_idx = min(range(len(offsets)), key=lambda index: abs(offsets[index]))
                sample_rows: list[dict[str, str]] = []
                for sequence_frame, offset in enumerate(offsets, start=1):
                    requested_time = center + offset
                    bounded_time = min(max(requested_time, 0.0), duration)
                    video_frame = min(frame_count - 1, max(0, round(bounded_time * fps)))
                    actual_time = video_frame / fps
                    sample_id = f"{event_id}_frame_{sequence_frame:03d}"
                    image_path = args.frames_dir / match / f"{sample_id}.jpg"
                    relative_image_path = image_path.relative_to(ROOT).as_posix()
                    saved = previous.get(sample_id, {})
                    image_status = "REUSED_FRAME_IMAGE" if image_path.is_file() else "EXTRACTED"

                    reused_row: dict[str, str] | None = None
                    if not saved.get("annotation_status"):
                        for candidate in old_by_match.get(match, []):
                            if abs(float(candidate["timestamp_seconds"]) - actual_time) <= REUSE_TOLERANCE_SECONDS:
                                reused_row = candidate
                                break
                    if reused_row:
                        reused += 1
                    annotation_status = saved.get("annotation_status", "")
                    if not annotation_status:
                        annotation_status = "completed" if reused_row else "pending"
                    source = saved.get("annotation_source", "")
                    if not source:
                        source = ("REUSED_EXISTING_MANUAL_GT:" + reused_row["event_id"]) if reused_row else ""

                    row = {
                        "sequence_id": event_id,
                        "sequence_frame": str(sequence_frame),
                        "sample_id": sample_id,
                        "event_id": event_id,
                        "is_event_frame": str(sequence_frame - 1 == event_idx),
                        "match_id": match,
                        "video_filename": video_name,
                        "capture_domain": "Veo/derivados",
                        "split": "train",
                        "timestamp_seconds": f"{actual_time:.6f}",
                        "timestamp_hhmmss": format_time(actual_time),
                        "image_path": relative_image_path,
                        "ball_visibility": reused_row.get("ball_visibility", "") if reused_row else saved.get("ball_visibility", ""),
                        "ball_of_play_box_json": reused_row.get("ball_of_play_box_json", "") if reused_row else saved.get("ball_of_play_box_json", ""),
                        "distractor_boxes_json": reused_row.get("distractor_boxes_json", "[]") if reused_row else saved.get("distractor_boxes_json", "[]"),
                        "goal_region_box_json": reused_row.get("goal_region_box_json", "") if reused_row else saved.get("goal_region_box_json", ""),
                        "goal_region_points_json": reused_row.get("goal_region_points_json", "") if reused_row else saved.get("goal_region_points_json", ""),
                        "goal_region_status": reused_row.get("goal_region_status", "") if reused_row else saved.get("goal_region_status", ""),
                        "goal_volume_points_json": reused_row.get("goal_volume_points_json", "") if reused_row else saved.get("goal_volume_points_json", ""),
                        "goal_volume_status": reused_row.get("goal_volume_status", "") if reused_row else saved.get("goal_volume_status", ""),
                        "annotation_status": annotation_status,
                        "annotator": "manual" if reused_row else saved.get("annotator", ""),
                        "annotation_source": source,
                        "notes": f"oracle_pair={pair['pair_id']}; role={role}; label={label}; offset_seconds={offset:+.1f}; image_status={image_status}",
                        "_target_video_frame": str(video_frame),
                        "_image_path_absolute": str(image_path),
                    }
                    sample_rows.append(row)

                # Seek once to the first missing sample and decode forward. Repeated random
                # seeks in long H.264 matches made the first preparation unnecessarily slow.
                missing: dict[int, list[dict[str, str]]] = {}
                for row in sample_rows:
                    image_path = Path(row["_image_path_absolute"])
                    if not image_path.is_file():
                        missing.setdefault(int(row["_target_video_frame"]), []).append(row)
                if missing:
                    targets = sorted(missing)
                    first, last = targets[0], targets[-1]
                    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                    current = first
                    while current <= last:
                        ok, image = capture.read()
                        if not ok or image is None:
                            raise RuntimeError(f"Falha ao decodificar {video_name} no frame {current}")
                        for row in missing.get(current, []):
                            image_path = Path(row["_image_path_absolute"])
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                                raise RuntimeError(f"Falha ao gravar imagem {image_path}")
                            generated += 1
                        current += 1
                for row in sample_rows:
                    row.pop("_target_video_frame")
                    row.pop("_image_path_absolute")
                results.extend(sample_rows)
                write_csv(args.output, results)
                completed_now = sum(item["annotation_status"] == "completed" for item in results)
                print(f"Preparado {pair['pair_id']} {role}: {len(sample_rows)} frames; imagens novas={generated}; anotados/reaproveitados={completed_now}", flush=True)
            finally:
                capture.release()

    # If the input manifest is extended later, retain annotations only for IDs in this frozen batch.
    write_csv(args.output, results)
    total = len(results)
    completed = sum(row["annotation_status"] == "completed" for row in results)
    event_rows = [row for row in results if row["is_event_frame"] == "True"]
    geometry_ready = sum(bool(row["goal_region_status"]) for row in event_rows)
    print(f"Pairs={len(pairs)}; sequencias={len(pairs)*2}; frames={total}")
    print(f"Novas imagens extraidas={generated}; anotacoes manuais reaproveitadas={reused}")
    print(f"Frames com bola anotada/com status={completed}/{total}; goleiras centrais classificadas={geometry_ready}/{len(event_rows)}")
    print(f"CSV retomavel: {args.output}")
    print(f"Imagens: {args.frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
