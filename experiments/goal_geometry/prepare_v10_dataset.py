"""Prepara dados de detector visual a partir de anotações quadrilaterais."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import yaml

from experiments.goal_geometry.run_v85_candidate_level import bbox
from experiments.goal_geometry.run_v86_proposal_generator import frame_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara os metadados do dataset V10.")
    parser.add_argument("--development-annotations", type=Path, required=True)
    parser.add_argument("--external-annotations", type=Path, required=True)
    parser.add_argument("--external-timestamps", type=Path, required=True)
    parser.add_argument("--external-frames-dir", type=Path, required=True)
    parser.add_argument("--additional-development-annotations", type=Path, nargs="+")
    parser.add_argument("--additional-frames-dir", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def group_rows(annotation_rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in annotation_rows:
        grouped.setdefault(row["frame_id"], []).append(row)
    return list(grouped.values())


def points_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return bbox(
        [
            (float(row[x]), float(row[y]))
            for x, y in [("lt_x", "lt_y"), ("rt_x", "rt_y"), ("lb_x", "lb_y"), ("rb_x", "rb_y")]
        ]
    )


def yolo_line(row: dict[str, str], width: int, height: int) -> str:
    left, top, right, bottom = points_box(row)
    center_x = ((left + right) / 2) / width
    center_y = ((top + bottom) / 2) / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    values = [center_x, center_y, box_width, box_height]
    if any(value <= 0.0 for value in values[2:]) or any(
        value < 0.0 or value > 1.0 for value in values
    ):
        raise ValueError(f"Caixa fora dos limites em {row['frame_id']}: {values}")
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def write_label(group: list[dict[str, str]], image_path: Path, label_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    label_path.parent.mkdir(parents=True, exist_ok=True)
    positive_rows = [row for row in group if row.get("original_label") != "not_goal"]
    label_path.write_text(
        "\n".join(yolo_line(row, width, height) for row in positive_rows)
        + ("\n" if positive_rows else ""),
        encoding="utf-8",
    )
    return {
        "frame_id": group[0]["frame_id"],
        "match_id": group[0]["match_id"],
        "image_path": str(image_path),
        "label_path": str(label_path),
        "width": width,
        "height": height,
        "structures": len(positive_rows),
        "visibility_policy": "annotated_visible_extent",
        "split": "by_match",
    }


def main() -> int:
    args = parse_args()
    development = rows(args.development_annotations)
    for additional_path in args.additional_development_annotations or []:
        development.extend(rows(additional_path))
    external = rows(args.external_annotations)
    timestamps = {
        row["match_id"] + "_" + f"{float(row['timestamp_seconds']):010.3f}": row
        for row in rows(args.external_timestamps)
    }
    args.output.mkdir(parents=True, exist_ok=True)
    development_labels = args.output / "labels" / "development"
    external_labels = args.output / "labels" / "external"
    development_manifest: list[dict[str, Any]] = []
    external_manifest: list[dict[str, Any]] = []
    for group in group_rows(development):
        row = group[0]
        image_path = Path(row["image_path"])
        if not row.get("image_path") or not image_path.exists():
            if not args.additional_frames_dir:
                raise FileNotFoundError(image_path)
            frame_name = f"{row['frame_id']}.jpg"
            candidate_paths = [
                frames_dir / category / frame_name
                for frames_dir in args.additional_frames_dir or []
                for category in ["pending", "not_goal", "goal_structure"]
            ]
            image_path = next(
                (candidate for candidate in candidate_paths if candidate.exists()),
                candidate_paths[0],
            )
        development_manifest.append(
            write_label(group, image_path, development_labels / f"{row['frame_id']}.txt")
        )
    for group in group_rows(external):
        row = group[0]
        image_path = frame_path(args.external_frames_dir, timestamps[row["frame_id"]])
        external_manifest.append(
            write_label(group, image_path, external_labels / f"{row['frame_id']}.txt")
        )
    all_matches = sorted({row["match_id"] for row in development_manifest})
    folds = [
        {
            "test_match": test_match,
            "train_matches": [match for match in all_matches if match != test_match],
        }
        for test_match in all_matches
    ]
    manifest_fields = list(development_manifest[0])
    with (args.output / "development_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(development_manifest)
    with (args.output / "external_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(external_manifest[0]))
        writer.writeheader()
        writer.writerows(external_manifest)
    (args.output / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(args.output.resolve()),
                "nc": 1,
                "names": ["goal_structure"],
                "note": (
                    "Images permanecem referenciadas pelos manifestos; "
                    "não são duplicadas nesta etapa."
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": args.output.name,
        "version": "V10_dataset_preparation",
        "class_names": ["goal_structure"],
        "development_frames": len(development_manifest),
        "development_structures": sum(row["structures"] for row in development_manifest),
        "development_negative_frames": sum(
            row["structures"] == 0 for row in development_manifest
        ),
        "external_frames": len(external_manifest),
        "external_structures": sum(row["structures"] for row in external_manifest),
        "matches": all_matches,
        "folds": folds,
        "source_resolution": "1920x1080 esperado e verificado por manifesto",
        "partial_visibility_policy": (
            "use the annotated visible extent without inventing occluded points"
        ),
        "negative_frames": "ainda não incluídos; serão ampliados com negativos difíceis",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
