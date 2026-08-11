"""Audit whether V8.5 positive frames contain the correctly localized goal."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from experiments.goal_geometry.run_geometry_versions import frame_path
from experiments.goal_geometry.run_v85_candidate_level import (
    bbox,
    bbox_iou,
    candidate_box,
    candidate_features,
    frame_candidates,
    models,
)

FEATURE_EXCLUDE = {"sample_id", "frame_id", "match_id", "label", "candidate_iou"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spatial audit of V8.5 positives.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--training-candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument("--partial-iou", type=float, default=0.20)
    parser.add_argument("--correct-iou", type=float, default=0.50)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    args = parse_args()
    annotations = read_rows(args.annotations)
    if any(row["annotation_status"] != "completed" for row in annotations):
        raise ValueError("As 15 anotações externas precisam estar completed")
    timestamps = {
        row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}": row
        for row in read_rows(args.timestamps)
    }
    training = read_rows(args.training_candidates)
    features = [key for key in training[0] if key not in FEATURE_EXCLUDE]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_overlays = args.output / "overlays"
    output_overlays.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for annotation in annotations:
        frame_id = annotation["frame_id"]
        timestamp = timestamps[frame_id]
        image_path = frame_path(args.frames_dir, timestamp)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        truth = bbox(
            [
                (float(annotation[x]), float(annotation[y]))
                for x, y in [("lt_x", "lt_y"), ("rt_x", "rt_y"), ("lb_x", "lb_y"), ("rb_x", "rb_y")]
            ]
        )
        candidates = frame_candidates(image, config)
        rows = []
        for index, candidate in enumerate(candidates):
            rows.append(
                {
                    "sample_id": f"{frame_id}_c{index + 1:03d}",
                    "frame_id": frame_id,
                    "match_id": annotation["match_id"],
                    **candidate_features(candidate, image.shape[1], image.shape[0]),
                    "candidate": candidate,
                }
            )
        train = [row for row in training if row["match_id"] != annotation["match_id"]]
        estimator = models()[args.model]
        estimator.fit(
            np.array([[float(row[feature]) for feature in features] for row in train]),
            np.array([row["label"] for row in train]),
        )
        if rows:
            probabilities = estimator.predict_proba(
                np.array([[row[feature] for feature in features] for row in rows])
            )[:, list(estimator.classes_).index("goal_structure")]
            best_index = int(np.argmax(probabilities))
            best = rows[best_index]
            score = float(probabilities[best_index])
            predicted_box = candidate_box(best["candidate"])
            iou = bbox_iou(truth, predicted_box)
        else:
            best = None
            score = 0.0
            predicted_box = (0.0, 0.0, 0.0, 0.0)
            iou = 0.0
        if iou >= args.correct_iou:
            localization_class = "CORRECT_LOCALIZATION"
        elif iou >= args.partial_iou:
            localization_class = "PARTIAL_LOCALIZATION"
        else:
            localization_class = "WRONG_LOCALIZATION"
        results.append(
            {
                "frame_id": frame_id,
                "match_id": annotation["match_id"],
                "true_label": annotation["label"],
                "frame_prediction": True,
                "candidate_score": score,
                "candidate_x1": predicted_box[0],
                "candidate_y1": predicted_box[1],
                "candidate_x2": predicted_box[2],
                "candidate_y2": predicted_box[3],
                "manual_x1": truth[0],
                "manual_y1": truth[1],
                "manual_x2": truth[2],
                "manual_y2": truth[3],
                "iou": iou,
                "localization_class": localization_class,
                "review_notes": "",
            }
        )
        overlay = image.copy()
        cv2.rectangle(
            overlay, (int(truth[0]), int(truth[1])), (int(truth[2]), int(truth[3])), (0, 0, 255), 3
        )
        if best is not None:
            cv2.rectangle(
                overlay,
                (int(predicted_box[0]), int(predicted_box[1])),
                (int(predicted_box[2]), int(predicted_box[3])),
                (255, 0, 0),
                3,
            )
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 72), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"RED=manual BLUE=V8.5 {localization_class}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"IoU={iou:.3f} score={score:.3f} {frame_id}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        match_dir = output_overlays / annotation["match_id"]
        match_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(match_dir / f"{frame_id}.jpg"), overlay)
    fields = list(results[0])
    with (args.output / "localization_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    correct = sum(row["localization_class"] == "CORRECT_LOCALIZATION" for row in results)
    partial = sum(row["localization_class"] == "PARTIAL_LOCALIZATION" for row in results)
    wrong = sum(row["localization_class"] == "WRONG_LOCALIZATION" for row in results)
    metrics = {
        "frames": len(results),
        "correct_localizations": correct,
        "partial_localizations": partial,
        "wrong_localizations": wrong,
        "localization_recall": correct / len(results) if results else 0.0,
        "usable_localization_recall": (correct + partial) / len(results) if results else 0.0,
        "partial_iou": args.partial_iou,
        "correct_iou": args.correct_iou,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tiles = []
    for result in results:
        overlay_path = output_overlays / result["match_id"] / f"{result['frame_id']}.jpg"
        tile = cv2.imread(str(overlay_path))
        if tile is None:
            continue
        tile = cv2.resize(tile, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            f"{result['localization_class']} IoU={float(result['iou']):.2f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
        )
        tiles.append(tile)
    if tiles:
        columns = 3
        rows_count = (len(tiles) + columns - 1) // columns
        sheet = np.zeros((rows_count * 270, columns * 480, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row_index, column_index = divmod(index, columns)
            sheet[
                row_index * 270 : (row_index + 1) * 270,
                column_index * 480 : (column_index + 1) * 480,
            ] = tile
        cv2.imwrite(str(args.output / "spatial_audit_contact_sheet.jpg"), sheet)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
