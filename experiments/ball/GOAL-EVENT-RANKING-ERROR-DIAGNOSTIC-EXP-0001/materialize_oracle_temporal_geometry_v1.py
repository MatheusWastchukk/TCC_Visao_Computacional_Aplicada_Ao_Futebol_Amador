"""Materialize per-frame goal geometry from reviewed temporal anchors.

Propagated coordinates are estimates derived from reviewed anchors and ORB/RANSAC
homographies. They are not frame-level ground truth. Source annotations are never
modified; the output is a separate CSV with explicit provenance and statuses.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


EXP = Path(__file__).resolve().parent
DEFAULT_BASE = EXP / "oracle_frame_annotations_v1_annotated.csv"
DEFAULT_REVIEW = EXP / "oracle_geometry_camera_review_v1.csv"
DEFAULT_MOTION = EXP / "camera_stability_audit_v1" / "camera_motion_pairs_v1.csv"
DEFAULT_OUTPUT = EXP / "oracle_frame_annotations_v1_temporal_geometry.csv"
DEFAULT_SUMMARY = EXP / "oracle_temporal_geometry_materialization_v1.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def point_transform(points_json: str, matrix: np.ndarray, scale_x: float, scale_y: float) -> str:
    if not points_json:
        return ""
    points = np.asarray(json.loads(points_json), dtype=np.float32)
    points *= np.asarray([scale_x, scale_y], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(mapped).all() or np.max(np.abs(mapped)) > 100000:
        return ""
    mapped /= np.asarray([scale_x, scale_y], dtype=np.float32)
    return json.dumps(np.round(mapped, 2).tolist(), separators=(",", ":"))


def frame_scale(image_path: str) -> tuple[float, float]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    reduced_width = min(960, width)
    return reduced_width / width, round(height * reduced_width / width) / height


def geometry_inside_frame(points_json: str, image_path: str) -> str:
    if not points_json:
        return "False"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    points = json.loads(points_json)
    return str(all(0 <= x < width and 0 <= y < height for x, y in points))


def transform_between(
    sequence_id: str,
    source_frame: int,
    target_frame: int,
    transitions: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    if source_frame == target_frame:
        return np.eye(3, dtype=np.float64)
    if source_frame < target_frame:
        matrix = np.eye(3, dtype=np.float64)
        for to_frame in range(source_frame + 1, target_frame + 1):
            step = transitions.get((sequence_id, to_frame))
            if step is None:
                raise ValueError(f"Missing homography: {sequence_id} -> frame {to_frame}")
            matrix = step @ matrix
        return matrix

    forward = transform_between(sequence_id, target_frame, source_frame, transitions)
    try:
        return np.linalg.inv(forward)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"Non-invertible homography chain: {sequence_id} {source_frame}->{target_frame}") from error


def make_anchors(
    base_rows: list[dict[str, str]], review_rows: list[dict[str, str]]
) -> dict[str, list[dict[str, str]]]:
    anchors: dict[str, list[dict[str, str]]] = {}
    for row in base_rows:
        if row.get("is_event_frame", "").lower() == "true":
            anchors.setdefault(row["sequence_id"], []).append(
                {
                    "sequence_frame": row["sequence_frame"],
                    "region_points": row.get("goal_region_points_json", ""),
                    "region_status": row.get("goal_region_status", ""),
                    "volume_points": row.get("goal_volume_points_json", ""),
                    "volume_status": row.get("goal_volume_status", ""),
                    "region_source": "MANUAL_CENTER",
                    "volume_source": "MANUAL_CENTER",
                    "image_path": row["image_path"],
                }
            )

    for row in review_rows:
        decision = row.get("review_decision", "")
        if not decision:
            raise ValueError(f"Pending geometry review: {row['sequence_id']} frame {row['sequence_frame']}")
        if decision in {"HOMOGRAPHY_ACCEPTED", "MANUAL_KEYFRAME", "MANUAL_KEYFRAME_REGION_ONLY"}:
            region_source = "REVIEWED_HOMOGRAPHY_ANCHOR" if decision == "HOMOGRAPHY_ACCEPTED" else "MANUAL_KEYFRAME"
            volume_source = region_source
        elif decision in {"NOT_VISIBLE", "UNCERTAIN"}:
            region_source = "REVIEWED_STATUS_ANCHOR"
            volume_source = "REVIEWED_STATUS_ANCHOR"
        else:
            raise ValueError(f"Unknown review decision {decision!r}")
        anchors.setdefault(row["sequence_id"], []).append(
            {
                "sequence_frame": row["sequence_frame"],
                "region_points": row.get("reviewed_goal_region_points_json", ""),
                "region_status": row.get("reviewed_goal_region_status", ""),
                "volume_points": row.get("reviewed_goal_volume_points_json", ""),
                "volume_status": row.get("reviewed_goal_volume_status", ""),
                "region_source": region_source,
                "volume_source": volume_source,
                "image_path": row["image_path"],
            }
        )
    for sequence_id in anchors:
        anchors[sequence_id].sort(key=lambda row: int(row["sequence_frame"]))
    return anchors


def materialize(base_path: Path, review_path: Path, motion_path: Path, output_path: Path) -> tuple[list[dict[str, str]], dict[str, Counter[str]]]:
    base_rows = read_csv(base_path)
    review_rows = read_csv(review_path)
    motion_rows = read_csv(motion_path)
    transitions = {
        (row["sequence_id"], int(row["to_frame"])): np.asarray(json.loads(row["homography_json"]), dtype=np.float64)
        for row in motion_rows
        if row.get("status") == "ESTIMATED" and row.get("homography_json")
    }
    anchors_by_sequence = make_anchors(base_rows, review_rows)

    extra_fields = [
        "effective_goal_region_points_json", "effective_goal_region_status", "goal_region_geometry_source",
        "goal_region_anchor_frame", "goal_region_anchor_status", "goal_region_geometry_inside_frame", "goal_region_geometry_notes",
        "effective_goal_volume_points_json", "effective_goal_volume_status", "goal_volume_geometry_source",
        "goal_volume_anchor_frame", "goal_volume_anchor_status", "goal_volume_geometry_inside_frame", "goal_volume_geometry_notes",
    ]
    output_rows: list[dict[str, str]] = []
    summary: dict[str, Counter[str]] = {"region_status": Counter(), "region_source": Counter(), "volume_status": Counter(), "volume_source": Counter()}
    for row in base_rows:
        sequence_id = row["sequence_id"]
        target_frame = int(row["sequence_frame"])
        anchors = anchors_by_sequence.get(sequence_id, [])
        if not anchors:
            raise ValueError(f"No geometry anchors for {sequence_id}")
        # Ties select the earlier reviewed anchor, making the result deterministic.
        anchor = min(anchors, key=lambda item: (abs(int(item["sequence_frame"]) - target_frame), int(item["sequence_frame"])))
        anchor_frame = int(anchor["sequence_frame"])
        is_anchor = anchor_frame == target_frame
        transform = transform_between(sequence_id, anchor_frame, target_frame, transitions)
        scale_x, scale_y = frame_scale(row["image_path"])
        result = dict(row)
        notes: list[str] = []

        for kind in ("region", "volume"):
            status = anchor[f"{kind}_status"]
            points = anchor[f"{kind}_points"]
            source = anchor[f"{kind}_source"] if is_anchor else f"HOMOGRAPHY_PROPAGATED_FROM_{anchor[f'{kind}_source']}"
            if status == "VALID" and points:
                mapped = points if is_anchor else point_transform(points, transform, scale_x, scale_y)
                if mapped:
                    effective_status = "VALID"
                else:
                    mapped = ""
                    effective_status = "UNCERTAIN"
                    notes.append(f"{kind}: transformação não finita; marcado UNCERTAIN")
            elif status:
                mapped = ""
                effective_status = status
                if not is_anchor:
                    notes.append(f"{kind}: status {status} herdado do frame âncora {anchor_frame}")
            else:
                mapped = ""
                effective_status = "UNCERTAIN"
                notes.append(f"{kind}: âncora sem geometria/status válido")

            result[f"effective_goal_{kind}_points_json"] = mapped
            result[f"effective_goal_{kind}_status"] = effective_status
            result[f"goal_{kind}_geometry_source"] = source
            result[f"goal_{kind}_anchor_frame"] = str(anchor_frame)
            result[f"goal_{kind}_anchor_status"] = status
            result[f"goal_{kind}_geometry_inside_frame"] = geometry_inside_frame(mapped, row["image_path"])
            result[f"goal_{kind}_geometry_notes"] = "; ".join(notes)
            summary[f"{kind}_status"][effective_status] += 1
            summary[f"{kind}_source"][source] += 1

        if notes:
            result["goal_region_geometry_notes"] = "; ".join(notes)
            result["goal_volume_geometry_notes"] = "; ".join(notes)
        output_rows.append(result)

    write_csv(output_path, output_rows, list(base_rows[0]) + extra_fields)
    return output_rows, summary


def write_summary(path: Path, rows: list[dict[str, str]], summary: dict[str, Counter[str]], review_rows: list[dict[str, str]]) -> None:
    decisions = Counter(row["review_decision"] for row in review_rows)
    lines = [
        "# Materialização da geometria temporal oracle — versão 1",
        "",
        "**Escopo:** 40 sequências TRAIN, 1.640 frames anotados a 5 FPS; somente os 20 pares congelados do piloto.",
        "**Entrada:** anotações BALL/geometria central e fila temporal revisada.",
        "**Saída:** `oracle_frame_annotations_v1_temporal_geometry.csv`.",
        "",
        "## Revisão de âncoras",
        "",
        f"- Total: {len(review_rows)}/320 decisões preenchidas.",
        *[f"- `{key}`: {value}" for key, value in sorted(decisions.items())],
        "",
        "## Geometria efetiva por frame",
        "",
        f"- Linhas materializadas: {len(rows)}.",
        *[f"- Abertura `{key}`: {value}" for key, value in sorted(summary["region_status"].items())],
        *[f"- Volume `{key}`: {value}" for key, value in sorted(summary["volume_status"].items())],
        "",
        "### Proveniência da abertura",
        "",
        *[f"- `{key}`: {value}" for key, value in sorted(summary["region_source"].items())],
        "",
        "### Proveniência do volume",
        "",
        *[f"- `{key}`: {value}" for key, value in sorted(summary["volume_source"].items())],
        "",
        "## Método e limites",
        "",
        "Cada frame recebe a âncora revisada mais próxima na mesma sequência; em empate, usa-se a âncora anterior. Coordenadas válidas são projetadas pela composição das homografias adjacentes ORB/RANSAC; âncoras manuais são mantidas diretamente. `NOT_VISIBLE`, `UNCERTAIN` e `UNCERTAIN_INHERITED` propagam-se como status sem criar pontos. A geometria resultante é estimativa temporal quando a origem começa com `HOMOGRAPHY_PROPAGATED`; não é ground truth anotado em cada frame.",
        "",
        "As anotações-fonte permanecem intactas. Não houve extração de features nem comparação de scores nesta etapa. A janela de incerteza conservadora ao redor dos status âncora deve ser examinada antes da análise oracle. Os dois volumes centrais cortados permanecem incertos e não foram completados por estimativa manual.",
        "",
        "A geometria é 2D projetada e não comprova cruzamento físico 3D da linha. O piloto permanece TRAIN-only; `VALIDATION` e `TEST` não foram usados.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--motion-pairs", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    rows, counts = materialize(args.base, args.review, args.motion_pairs, args.output)
    write_summary(args.summary, rows, counts, read_csv(args.review))
    print(f"Frames materializados: {len(rows)}")
    for name, counter in counts.items():
        print(f"{name}: {dict(counter)}")
    print(f"CSV: {args.output}")
    print(f"Resumo: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
