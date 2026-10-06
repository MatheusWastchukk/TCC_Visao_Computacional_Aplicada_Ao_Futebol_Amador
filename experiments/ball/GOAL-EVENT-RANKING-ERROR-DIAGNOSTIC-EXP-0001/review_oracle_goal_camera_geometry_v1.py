"""Revisa geometrias da goleira projetadas entre frames e permite criar keyframes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


EXP = Path(__file__).resolve().parent
DEFAULT_INPUT = EXP / "oracle_frame_annotations_v1_annotated.csv"
DEFAULT_MOTION = EXP / "camera_stability_audit_v1" / "camera_motion_pairs_v1.csv"
DEFAULT_OUTPUT = EXP / "oracle_geometry_camera_review_v1.csv"
SAMPLE_INDICES = (0, 10, 15, 18, 22, 25, 30, 40)
REGION_POINTS = 4
VOLUME_POINTS = 6
REVIEW_FIELDS = [
    "sequence_id", "event_id", "match_id", "sequence_frame", "timestamp_hhmmss", "image_path",
    "ball_visibility", "ball_of_play_box_json",
    "review_scope", "reference_volume_status", "proposed_goal_region_points_json",
    "proposed_goal_volume_points_json", "review_decision", "reviewed_goal_region_points_json",
    "reviewed_goal_region_status", "reviewed_goal_volume_points_json", "reviewed_goal_volume_status", "notes",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def project_points(points_json: str, transform: np.ndarray, scale_x: float, scale_y: float) -> str:
    if not points_json:
        return ""
    points = np.array(json.loads(points_json), dtype=np.float32)
    points *= np.array([scale_x, scale_y], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points.reshape(1, -1, 2), transform).reshape(-1, 2)
    if not np.isfinite(mapped).all() or np.max(np.abs(mapped)) > 100000:
        return ""
    mapped /= np.array([scale_x, scale_y], dtype=np.float32)
    return json.dumps(np.round(mapped, 2).tolist(), separators=(",", ":"))


def prepare_rows(input_path: Path, motion_path: Path, output_path: Path) -> list[dict[str, str]]:
    source_rows = load_csv(input_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault(row["sequence_id"], []).append(row)
    for frames in grouped.values():
        frames.sort(key=lambda row: int(row["sequence_frame"]))

    motion = load_csv(motion_path)
    transitions: dict[tuple[str, int], np.ndarray] = {}
    for row in motion:
        if row.get("status") == "ESTIMATED" and row.get("homography_json"):
            transitions[(row["sequence_id"], int(row["to_frame"]))] = np.array(json.loads(row["homography_json"]), dtype=np.float64)

    existing: dict[tuple[str, str], dict[str, str]] = {}
    if output_path.exists():
        existing = {(row["sequence_id"], row["sequence_frame"]): row for row in load_csv(output_path)}

    prepared: list[dict[str, str]] = []
    for sequence_id, frames in sorted(grouped.items()):
        center_index = next((i for i, row in enumerate(frames) if row.get("is_event_frame") == "True"), len(frames) // 2)
        center = frames[center_index]
        center_region = center.get("goal_region_points_json", "")
        center_volume = center.get("goal_volume_points_json", "") if center.get("goal_volume_status") == "VALID" else ""
        ref_volume_status = center.get("goal_volume_status", "")

        for frame_index in SAMPLE_INDICES:
            if frame_index == center_index or frame_index >= len(frames):
                continue
            frame = frames[frame_index]
            transform: np.ndarray | None = np.eye(3, dtype=np.float64)
            if frame_index > center_index:
                for next_index in range(center_index + 1, frame_index + 1):
                    matrix = transitions.get((sequence_id, int(frames[next_index]["sequence_frame"])))
                    if matrix is None:
                        transform = None
                        break
                    transform = matrix @ transform
            else:
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

            proposal_region = ""
            proposal_volume = ""
            if transform is not None:
                image = cv2.imread(frame["image_path"], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(frame["image_path"])
                reduced_width = min(960, image.shape[1])
                scale_x = reduced_width / image.shape[1]
                scale_y = round(image.shape[0] * reduced_width / image.shape[1]) / image.shape[0]
                proposal_region = project_points(center_region, transform, scale_x, scale_y)
                proposal_volume = project_points(center_volume, transform, scale_x, scale_y)

            key = (sequence_id, frame["sequence_frame"])
            prior = existing.get(key, {})
            inherited_note = ""
            if ref_volume_status == "UNCERTAIN":
                inherited_note = "Volume central UNCERTAIN; esta fila não solicita nova marcação de volume."
            row = {field: "" for field in REVIEW_FIELDS}
            row.update(
                {
                    "sequence_id": sequence_id,
                    "event_id": frame["event_id"],
                    "match_id": frame["match_id"],
                    "sequence_frame": frame["sequence_frame"],
                    "timestamp_hhmmss": frame["timestamp_hhmmss"],
                    "image_path": frame["image_path"],
                    "ball_visibility": frame.get("ball_visibility", ""),
                    "ball_of_play_box_json": frame.get("ball_of_play_box_json", ""),
                    "review_scope": "TEMPORAL_KEYFRAME_CANDIDATE",
                    "reference_volume_status": ref_volume_status,
                    "proposed_goal_region_points_json": proposal_region,
                    "proposed_goal_volume_points_json": proposal_volume,
                    "notes": inherited_note,
                }
            )
            for field in REVIEW_FIELDS:
                if prior.get("review_decision"):
                    row[field] = prior.get(field, row[field])
                elif prior.get(field) and field not in {"proposed_goal_region_points_json", "proposed_goal_volume_points_json"}:
                    row[field] = prior[field]
            prepared.append(row)
    save_csv(output_path, prepared)
    return prepared


def draw_polygon(canvas: np.ndarray, points_json: str, color: tuple[int, int, int], label: str) -> None:
    if not points_json:
        return
    points = np.array(json.loads(points_json), dtype=np.int32)
    cv2.polylines(canvas, [points], True, color, 2)
    x, y = points[0]
    cv2.putText(canvas, label, (int(x), max(24, int(y) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--motion-pairs", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()

    rows = prepare_rows(args.input, args.motion_pairs, args.output)
    pending = [row for row in rows if not row.get("review_decision")]
    print(f"Fila preparada: {len(rows)} frames temporais; pendentes={len(pending)}; arquivo={args.output}")
    if args.prepare_only or not pending:
        return 0

    queue = pending if args.pending_only else rows
    queue = [row for row in queue if not row.get("review_decision")]
    window = "Revisão da geometria temporal da goleira"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state: dict[str, object] = {"points": [], "scale": 1.0, "phase": "DECISION", "row": None}

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        phase = str(state["phase"])
        limit = REGION_POINTS if phase == "REGION" else VOLUME_POINTS if phase == "VOLUME" else 0
        points = state["points"]
        if limit and len(points) < limit:
            scale = float(state["scale"])
            points.append((round(x / scale), round(y / scale)))

    cv2.setMouseCallback(window, mouse)
    for index, row in enumerate(queue, start=1):
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        scale = min(1.0, args.max_preview_width / image.shape[1])
        resume_phase = (
            "VOLUME"
            if row.get("reviewed_goal_region_status") == "VALID"
            and row.get("reference_volume_status") == "VALID"
            and not row.get("reviewed_goal_volume_points_json")
            else "DECISION"
        )
        state.update({"points": [], "scale": scale, "phase": resume_phase, "row": row})
        while True:
            canvas = image.copy()
            draw_polygon(canvas, row.get("proposed_goal_region_points_json", ""), (0, 140, 255), "ABERTURA PROJETADA")
            draw_polygon(canvas, row.get("proposed_goal_volume_points_json", ""), (255, 160, 0), "REDE PROJETADA")
            if row.get("reviewed_goal_region_points_json"):
                draw_polygon(canvas, row["reviewed_goal_region_points_json"], (0, 255, 255), "KEYFRAME MANUAL")
            if row.get("ball_of_play_box_json"):
                x1, y1, x2, y2 = (round(float(v)) for v in json.loads(row["ball_of_play_box_json"]))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
            points = state["points"]
            if points:
                color = (0, 255, 255) if state["phase"] == "REGION" else (255, 160, 0)
                cv2.polylines(canvas, [np.array(points, dtype=np.int32)], len(points) >= 3, color, 2)
                for point_number, point in enumerate(points, start=1):
                    cv2.circle(canvas, point, 5, color, -1)
                    cv2.putText(canvas, str(point_number), (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 1250), 96), (0, 0, 0), -1)
            cv2.putText(canvas, f"{index}/{len(queue)} | {row['sequence_id']} | frame {row['sequence_frame']} | {row['timestamp_hhmmss']}", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            if state["phase"] == "DECISION":
                help_text = "A aceitar projeção | K keyframe manual | N não visível | U incerto | P adiar | Q salvar e sair"
            elif state["phase"] == "REGION":
                help_text = "KEYFRAME: clique os 4 cantos da abertura em ordem; G confirma | C limpa | Q salva e sai"
            else:
                help_text = "KEYFRAME: clique 6 pontos do contorno da rede em ordem; G confirma | C limpa | Q salva e sai"
            cv2.putText(canvas, help_text, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 120), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Laranja = abertura proposta | Azul = rede proposta | Verde = bola anotada", (16, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 230, 255), 1, cv2.LINE_AA)
            rendered = cv2.resize(canvas, None, fx=scale, fy=scale) if scale != 1.0 else canvas
            cv2.imshow(window, rendered)
            key = cv2.waitKeyEx(50)
            if key in {-1, 255}:
                continue
            char = chr(key & 0xFF).lower()
            if char == "q" or key == 27:
                save_csv(args.output, rows)
                cv2.destroyAllWindows()
                print(f"Progresso salvo: {args.output}")
                return 0
            if char == "c" and state["phase"] in {"REGION", "VOLUME"}:
                state["points"] = []
                continue
            if state["phase"] == "DECISION":
                if char == "a":
                    if not row.get("proposed_goal_region_points_json"):
                        print("Sem proposta de abertura; use K para keyframe manual ou U/N.")
                        continue
                    row["review_decision"] = "HOMOGRAPHY_ACCEPTED"
                    row["reviewed_goal_region_points_json"] = row["proposed_goal_region_points_json"]
                    row["reviewed_goal_region_status"] = "VALID"
                    if row.get("proposed_goal_volume_points_json"):
                        row["reviewed_goal_volume_points_json"] = row["proposed_goal_volume_points_json"]
                        row["reviewed_goal_volume_status"] = "VALID"
                    elif row.get("reference_volume_status") == "UNCERTAIN":
                        row["reviewed_goal_volume_status"] = "UNCERTAIN_INHERITED"
                    elif row.get("reference_volume_status") == "VALID":
                        row["reviewed_goal_volume_status"] = "UNCERTAIN"
                    save_csv(args.output, rows)
                    break
                if char == "k":
                    state.update({"phase": "REGION", "points": []})
                    continue
                if char == "n":
                    row["review_decision"] = "NOT_VISIBLE"
                    row["reviewed_goal_region_status"] = "NOT_VISIBLE"
                    row["reviewed_goal_volume_status"] = "UNCERTAIN_INHERITED" if row.get("reference_volume_status") == "UNCERTAIN" else "NOT_VISIBLE"
                    save_csv(args.output, rows)
                    break
                if char == "u":
                    row["review_decision"] = "UNCERTAIN"
                    row["reviewed_goal_region_status"] = "UNCERTAIN"
                    row["reviewed_goal_volume_status"] = "UNCERTAIN_INHERITED" if row.get("reference_volume_status") == "UNCERTAIN" else "UNCERTAIN"
                    save_csv(args.output, rows)
                    break
                if char == "p":
                    break
            elif char == "g" and state["phase"] == "REGION":
                if len(points) != REGION_POINTS:
                    print(f"Clique exatamente {REGION_POINTS} pontos para a abertura.")
                    continue
                row["reviewed_goal_region_points_json"] = json.dumps(points)
                row["reviewed_goal_region_status"] = "VALID"
                if row.get("reference_volume_status") == "VALID":
                    save_csv(args.output, rows)
                    state.update({"phase": "VOLUME", "points": []})
                    continue
                row["review_decision"] = "MANUAL_KEYFRAME_REGION_ONLY"
                row["reviewed_goal_volume_status"] = "UNCERTAIN_INHERITED"
                row["notes"] = (row.get("notes", "") + " Volume central incerto; não solicitado nesta revisão.").strip()
                save_csv(args.output, rows)
                break
            elif char == "g" and state["phase"] == "VOLUME":
                if len(points) != VOLUME_POINTS:
                    print(f"Clique exatamente {VOLUME_POINTS} pontos para o volume.")
                    continue
                front = np.array(json.loads(row["reviewed_goal_region_points_json"]), dtype=np.float32)
                outside = [p for p in points if cv2.pointPolygonTest(front, (float(p[0]), float(p[1])), False) < 0]
                if not outside:
                    print("Inclua ao menos um ponto fora do plano da abertura.")
                    continue
                row["reviewed_goal_volume_points_json"] = json.dumps(points)
                row["reviewed_goal_volume_status"] = "VALID"
                row["review_decision"] = "MANUAL_KEYFRAME"
                save_csv(args.output, rows)
                break
    cv2.destroyAllWindows()
    print(f"Revisão do lote concluída/salva: {args.output}")
    print(f"Pendentes: {sum(not row.get('review_decision') for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
