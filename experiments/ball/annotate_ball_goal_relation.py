"""Revisa por clipe a relevância e a qualidade da detecção de goleira para a trajetória."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


FIELDS = ["sequence_id", "match_id", "frames", "manual_relation_quality", "annotation_status"]


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def draw_box(image, box_json: str, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    if not box_json:
        return
    x1, y1, x2, y2 = (round(float(value)) for value in json.loads(box_json))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Revisa a relação visual entre trajetória da bola e goleira.")
    parser.add_argument("--relation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    with args.relation.open(newline="", encoding="utf-8") as file:
        relation_rows = list(csv.DictReader(file))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in relation_rows:
        grouped.setdefault(row["sequence_id"], []).append(row)
    existing: dict[str, dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["sequence_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(sequence_id, {
        "sequence_id": sequence_id, "match_id": frames[0]["match_id"], "frames": str(len(frames)),
        "manual_relation_quality": "", "annotation_status": "pending",
    }) for sequence_id, frames in sorted(grouped.items())]
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print(f"Não há clipes pendentes: {args.output}")
        return 0

    window = "Revisão: trajetória x goleira"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    labels = {
        "r": "RELEVANTE_CORRETA", "f": "FALHA_DETECCAO", "a": "SEM_GOLEIRA_RELEVANTE", "u": "INCONCLUSIVO",
    }
    for clip_index, review in enumerate(pending, start=1):
        frames = sorted(grouped[review["sequence_id"]], key=lambda row: int(row["sequence_frame"]))
        frame_index = 0
        while True:
            row = frames[frame_index]
            image = cv2.imread(row["image_path"])
            if image is None:
                raise FileNotFoundError(row["image_path"])
            draw_box(image, row.get("target_box_json", ""), (0, 220, 0), "ALVO")
            draw_box(image, row.get("selected_ball_box_json", ""), (0, 220, 255), "TRACKER", 3)
            draw_box(image, row.get("nearest_goal_box_json", ""), (255, 160, 0), "GOLEIRA")
            distance = row.get("ball_goal_center_distance") or "-"
            inside = row.get("ball_inside_nearest_goal") or "-"
            cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1000), 70), (0, 0, 0), -1)
            cv2.putText(image, f"Clipe {clip_index}/{len(pending)} | {review['sequence_id']} | quadro {frame_index + 1}/{len(frames)} | distancia={distance} | dentro={inside}", (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "A/D ou setas: navegar | R relevante/correta | F falha | A sem goleira relevante | U inconclusivo | Q sair", (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 120), 2, cv2.LINE_AA)
            scale = min(1.0, args.max_preview_width / image.shape[1])
            canvas = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(60)
            if key in {-1, 255}:
                continue
            char = chr(key & 0xFF).lower()
            if char == "q" or key == 27:
                save_csv(args.output, rows)
                cv2.destroyAllWindows()
                print(f"Progresso salvo: {args.output}")
                return 0
            if char == "a" or key == 2424832:
                frame_index = max(0, frame_index - 1)
                continue
            if char == "d" or key == 2555904:
                frame_index = min(len(frames) - 1, frame_index + 1)
                continue
            if char in labels:
                review["manual_relation_quality"] = labels[char]
                review["annotation_status"] = "completed"
                save_csv(args.output, rows)
                print(f"Salvo: {review['sequence_id']} -> {labels[char]}")
                break
    cv2.destroyAllWindows()
    print(f"Revisão concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
