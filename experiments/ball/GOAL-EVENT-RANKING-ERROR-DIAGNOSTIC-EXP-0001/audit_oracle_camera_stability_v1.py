"""Audita movimento aparente de câmera nas sequências anotadas do piloto oracle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


EXP = Path(__file__).resolve().parent
DEFAULT_INPUT = EXP / "oracle_frame_annotations_v1_annotated.csv"
DEFAULT_OUTPUT_DIR = EXP / "camera_stability_audit_v1"
SAMPLE_INDICES = (0, 10, 20, 30, 40)
THUMBNAIL_SIZE = (320, 180)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def estimate_transform(previous: np.ndarray, current: np.ndarray, orb: cv2.ORB) -> dict[str, object]:
    height, width = previous.shape[:2]
    kp_a, desc_a = orb.detectAndCompute(previous, None)
    kp_b, desc_b = orb.detectAndCompute(current, None)
    result: dict[str, object] = {
        "keypoints_previous": len(kp_a),
        "keypoints_current": len(kp_b),
        "ratio_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "corner_motion_fraction": "",
        "scale_change_fraction": "",
        "homography_json": "",
        "status": "LOW_SUPPORT",
    }
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    result["ratio_matches"] = len(good)
    if len(good) < 8:
        return result

    points_a = np.float32([kp_a[match.queryIdx].pt for match in good])
    points_b = np.float32([kp_b[match.trainIdx].pt for match in good])
    matrix, inlier_mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 4.0)
    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        return result

    inliers = int(inlier_mask.ravel().sum())
    result["inliers"] = inliers
    result["inlier_ratio"] = round(inliers / len(good), 4)
    if inliers < 8 or inliers / len(good) < 0.20:
        return result
    result["homography_json"] = json.dumps(matrix.tolist(), separators=(",", ":"))

    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    mapped = cv2.perspectiveTransform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(mapped).all():
        return result
    motion = np.linalg.norm(mapped - corners, axis=1)
    result["corner_motion_fraction"] = round(float(np.median(motion) / math.hypot(width, height)), 6)
    projected_area = abs(float(cv2.contourArea(mapped.astype(np.float32))))
    original_area = float((width - 1) * (height - 1))
    scale = math.sqrt(projected_area / original_area) if original_area else 1.0
    result["scale_change_fraction"] = round(scale - 1.0, 6)
    result["status"] = "ESTIMATED"
    return result


def make_contact_sheets(sequence_rows: dict[str, list[dict[str, str]]], output_dir: Path) -> list[Path]:
    sequence_ids = sorted(sequence_rows)
    sheets: list[Path] = []
    for chunk_start in range(0, len(sequence_ids), 5):
        chunk = sequence_ids[chunk_start : chunk_start + 5]
        sheet = np.full((len(chunk) * (THUMBNAIL_SIZE[1] + 28), len(SAMPLE_INDICES) * THUMBNAIL_SIZE[0], 3), 28, dtype=np.uint8)
        for row_index, sequence_id in enumerate(chunk):
            frames = sequence_rows[sequence_id]
            for column_index, frame_index in enumerate(SAMPLE_INDICES):
                record = frames[min(frame_index, len(frames) - 1)]
                image = cv2.imread(record["image_path"])
                if image is None:
                    continue
                thumb = cv2.resize(image, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
                if record.get("is_event_frame") == "True":
                    region = record.get("goal_region_points_json", "")
                    volume = record.get("goal_volume_points_json", "")
                    for encoded, color in ((region, (0, 140, 255)), (volume, (255, 160, 0))):
                        if encoded:
                            points = np.array(json.loads(encoded), dtype=np.float32)
                            points *= np.array([THUMBNAIL_SIZE[0] / image.shape[1], THUMBNAIL_SIZE[1] / image.shape[0]])
                            cv2.polylines(thumb, [points.astype(np.int32)], True, color, 2)
                cv2.rectangle(thumb, (0, 0), (THUMBNAIL_SIZE[0] - 1, 23), (0, 0, 0), -1)
                caption = f"f{record['sequence_frame']}  {record['timestamp_hhmmss']}"
                cv2.putText(thumb, caption, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                x = column_index * THUMBNAIL_SIZE[0]
                y = row_index * (THUMBNAIL_SIZE[1] + 28)
                sheet[y + 28 : y + 28 + THUMBNAIL_SIZE[1], x : x + THUMBNAIL_SIZE[0]] = thumb
            y = row_index * (THUMBNAIL_SIZE[1] + 28)
            cv2.putText(sheet, sequence_id, (5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1, cv2.LINE_AA)
        path = output_dir / f"contact_sheet_{chunk_start // 5 + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets.append(path)
    return sheets


def make_goal_warp_review_sheets(
    sequence_rows: dict[str, list[dict[str, str]]],
    pair_rows: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    """Create visual-only proposals by warping center geometry across frames."""
    transitions: dict[tuple[str, int], np.ndarray] = {}
    for row in pair_rows:
        if row["status"] == "ESTIMATED" and row.get("homography_json"):
            transitions[(str(row["sequence_id"]), int(row["to_frame"]))] = np.array(json.loads(str(row["homography_json"])), dtype=np.float64)

    sequence_ids = sorted(sequence_rows)
    sample_indices = (0, 10, 15, 18, 20, 22, 25, 30, 40)
    sheets: list[Path] = []
    for chunk_start in range(0, len(sequence_ids), 5):
        chunk = sequence_ids[chunk_start : chunk_start + 5]
        sheet = np.full((len(chunk) * (THUMBNAIL_SIZE[1] + 28), len(sample_indices) * THUMBNAIL_SIZE[0], 3), 28, dtype=np.uint8)
        for row_index, sequence_id in enumerate(chunk):
            frames = sequence_rows[sequence_id]
            center_index = next((i for i, frame in enumerate(frames) if frame.get("is_event_frame") == "True"), len(frames) // 2)
            center = frames[center_index]
            for column_index, frame_index in enumerate(sample_indices):
                record = frames[min(frame_index, len(frames) - 1)]
                image = cv2.imread(record["image_path"])
                if image is None:
                    continue
                thumb = cv2.resize(image, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
                transform: np.ndarray | None = np.eye(3, dtype=np.float64)
                if frame_index > center_index:
                    for next_index in range(center_index + 1, frame_index + 1):
                        matrix = transitions.get((sequence_id, int(frames[next_index]["sequence_frame"])))
                        if matrix is None:
                            transform = None
                            break
                        transform = matrix @ transform
                elif frame_index < center_index:
                    forward = np.eye(3, dtype=np.float64)
                    for next_index in range(frame_index + 1, center_index + 1):
                        matrix = transitions.get((sequence_id, int(frames[next_index]["sequence_frame"])))
                        if matrix is None:
                            transform = None
                            break
                        forward = matrix @ forward
                    if transform is not None:
                        try:
                            transform = np.linalg.inv(forward)
                        except np.linalg.LinAlgError:
                            transform = None

                if transform is not None:
                    reduced_width = min(960, image.shape[1])
                    scale_x = reduced_width / image.shape[1]
                    scale_y = round(image.shape[0] * reduced_width / image.shape[1]) / image.shape[0]
                    display_scale = THUMBNAIL_SIZE[0] / reduced_width
                    for field, color in (("goal_region_points_json", (0, 140, 255)), ("goal_volume_points_json", (255, 160, 0))):
                        encoded = center.get(field, "")
                        if not encoded:
                            continue
                        points = np.array(json.loads(encoded), dtype=np.float32)
                        points *= np.array([scale_x, scale_y], dtype=np.float32)
                        mapped = cv2.perspectiveTransform(points.reshape(1, -1, 2), transform).reshape(-1, 2)
                        mapped *= display_scale
                        if np.isfinite(mapped).all() and np.max(np.abs(mapped)) < 10000:
                            cv2.polylines(thumb, [mapped.astype(np.int32)], True, color, 2)

                cv2.rectangle(thumb, (0, 0), (THUMBNAIL_SIZE[0] - 1, 23), (0, 0, 0), -1)
                caption = f"f{record['sequence_frame']} {record['timestamp_hhmmss']}"
                cv2.putText(thumb, caption, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
                x = column_index * THUMBNAIL_SIZE[0]
                y = row_index * (THUMBNAIL_SIZE[1] + 28)
                sheet[y + 28 : y + 28 + THUMBNAIL_SIZE[1], x : x + THUMBNAIL_SIZE[0]] = thumb
            y = row_index * (THUMBNAIL_SIZE[1] + 28)
            cv2.putText(sheet, f"{sequence_id} | abertura e rede projetadas por homografia", (5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 230, 255), 1, cv2.LINE_AA)
        path = output_dir / f"goal_warp_review_{chunk_start // 5 + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        sheets.append(path)
    return sheets


def summarize_event_windows(
    sequence_rows: dict[str, list[dict[str, str]]],
    pair_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize apparent motion in the two seconds centered on each event."""
    summaries: list[dict[str, object]] = []
    for sequence_id, frames in sorted(sequence_rows.items()):
        event = next((row for row in frames if row.get("is_event_frame") == "True"), None)
        if event is None:
            continue
        event_frame = int(event["sequence_frame"])
        start_frame = event_frame - 5
        end_frame = event_frame + 5
        transitions = [
            row
            for row in pair_rows
            if row["sequence_id"] == sequence_id
            and start_frame <= int(row["from_frame"]) < end_frame
        ]
        motions = [float(row["corner_motion_fraction"]) for row in transitions if row["corner_motion_fraction"] != ""]
        summaries.append(
            {
                "sequence_id": sequence_id,
                "match_id": event["match_id"],
                "event_id": event["event_id"],
                "event_frame": event_frame,
                "from_frame_inclusive": start_frame,
                "to_frame_inclusive": end_frame,
                "transitions": len(transitions),
                "estimated_transitions": len(motions),
                "low_support_transitions": len(transitions) - len(motions),
                "cumulative_corner_motion_fraction": round(sum(motions), 6),
                "median_corner_motion_fraction": round(float(np.median(motions)), 6) if motions else "",
                "p90_corner_motion_fraction": round(float(np.percentile(motions, 90)), 6) if motions else "",
                "max_corner_motion_fraction": round(max(motions), 6) if motions else "",
                "path_gt_005_diagonal": sum(motions) > 0.05,
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.input):
        grouped[row["sequence_id"]].append(row)
    for sequence_id in grouped:
        grouped[sequence_id].sort(key=lambda row: int(row["sequence_frame"]))

    cv2.setNumThreads(2)
    orb = cv2.ORB_create(nfeatures=1600, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    pair_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for sequence_id, frames in sorted(grouped.items()):
        motions: list[float] = []
        scales: list[float] = []
        estimated = 0
        for index in range(1, len(frames)):
            previous = cv2.imread(frames[index - 1]["image_path"], cv2.IMREAD_GRAYSCALE)
            current = cv2.imread(frames[index]["image_path"], cv2.IMREAD_GRAYSCALE)
            if previous is None or current is None:
                raise FileNotFoundError(frames[index - 1 if previous is None else index]["image_path"])
            original_height, original_width = previous.shape
            target_width = min(960, original_width)
            target_height = round(original_height * target_width / original_width)
            previous = cv2.resize(previous, (target_width, target_height), interpolation=cv2.INTER_AREA)
            current = cv2.resize(current, (target_width, target_height), interpolation=cv2.INTER_AREA)
            estimate = estimate_transform(previous, current, orb)
            estimate.update(
                {
                    "sequence_id": sequence_id,
                    "from_frame": frames[index - 1]["sequence_frame"],
                    "to_frame": frames[index]["sequence_frame"],
                    "from_timestamp": frames[index - 1]["timestamp_hhmmss"],
                    "to_timestamp": frames[index]["timestamp_hhmmss"],
                }
            )
            pair_rows.append(estimate)
            if estimate["status"] == "ESTIMATED":
                estimated += 1
                motions.append(float(estimate["corner_motion_fraction"]))
                scales.append(float(estimate["scale_change_fraction"]))

        summaries.append(
            {
                "sequence_id": sequence_id,
                "match_id": frames[0]["match_id"],
                "event_id": frames[0]["event_id"],
                "frames": len(frames),
                "transitions": max(0, len(frames) - 1),
                "estimated_transitions": estimated,
                "low_support_transitions": max(0, len(frames) - 1) - estimated,
                "support_fraction": round(estimated / max(1, len(frames) - 1), 4),
                "median_corner_motion_fraction": round(float(np.median(motions)), 6) if motions else "",
                "p90_corner_motion_fraction": round(float(np.percentile(motions, 90)), 6) if motions else "",
                "max_corner_motion_fraction": round(max(motions), 6) if motions else "",
                "cumulative_corner_motion_fraction": round(sum(motions), 6) if motions else "",
                "median_scale_change_fraction": round(float(np.median(scales)), 6) if scales else "",
                "max_abs_scale_change_fraction": round(max((abs(value) for value in scales), default=0.0), 6) if scales else "",
            }
        )

    write_csv(
        args.output_dir / "camera_motion_pairs_v1.csv",
        pair_rows,
        ["sequence_id", "from_frame", "to_frame", "from_timestamp", "to_timestamp", "keypoints_previous", "keypoints_current", "ratio_matches", "inliers", "inlier_ratio", "corner_motion_fraction", "scale_change_fraction", "homography_json", "status"],
    )
    write_csv(
        args.output_dir / "camera_motion_summary_v1.csv",
        summaries,
        ["sequence_id", "match_id", "event_id", "frames", "transitions", "estimated_transitions", "low_support_transitions", "support_fraction", "median_corner_motion_fraction", "p90_corner_motion_fraction", "max_corner_motion_fraction", "cumulative_corner_motion_fraction", "median_scale_change_fraction", "max_abs_scale_change_fraction"],
    )
    event_windows = summarize_event_windows(grouped, pair_rows)
    write_csv(
        args.output_dir / "camera_motion_event_window_v1.csv",
        event_windows,
        ["sequence_id", "match_id", "event_id", "event_frame", "from_frame_inclusive", "to_frame_inclusive", "transitions", "estimated_transitions", "low_support_transitions", "cumulative_corner_motion_fraction", "median_corner_motion_fraction", "p90_corner_motion_fraction", "max_corner_motion_fraction", "path_gt_005_diagonal"],
    )
    sheets = make_contact_sheets(grouped, args.output_dir)
    warp_sheets = make_goal_warp_review_sheets(grouped, pair_rows, args.output_dir)
    print(f"Sequências: {len(summaries)}; transições estimadas: {sum(int(row['estimated_transitions']) for row in summaries)}/{sum(int(row['transitions']) for row in summaries)}")
    print(f"Resumo: {args.output_dir / 'camera_motion_summary_v1.csv'}")
    print(f"Pares: {args.output_dir / 'camera_motion_pairs_v1.csv'}")
    print(f"Folhas de contato: {len(sheets)} em {args.output_dir}")
    print(f"Folhas de revisão das geometrias projetadas: {len(warp_sheets)}")
    for row in sorted(summaries, key=lambda item: float(item["p90_corner_motion_fraction"] or 0), reverse=True)[:10]:
        print(f"{row['sequence_id']}: support={row['support_fraction']} p90_motion={row['p90_corner_motion_fraction']} max={row['max_corner_motion_fraction']} cumulative={row['cumulative_corner_motion_fraction']} low_support={row['low_support_transitions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
