"""Marca a projeção visual do volume interno da rede no frame decisivo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


EXTRA_FIELDS = ["goal_volume_points_json", "goal_volume_status"]


def save_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_polygon(image, points_json: str, color: tuple[int, int, int], label: str) -> None:
    if not points_json:
        return
    points = np.array(json.loads(points_json), dtype="int32")
    cv2.polylines(image, [points], True, color, 2)
    x, y = points[0]
    cv2.putText(image, label, (int(x), max(24, int(y) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Marca o volume projetado da rede no frame decisivo.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    output = args.output or args.input
    with args.input.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
        for row in rows:
            row.setdefault(field, "")
    candidates = [row for row in rows if row.get("is_event_frame") == "True" and row.get("goal_region_status") == "VALID"]
    pending = [row for row in candidates if not args.pending_only or not row.get("goal_volume_status")]
    if not pending:
        print(f"Não há volumes pendentes: {output}")
        return 0

    window = "Marcação do volume projetado da rede"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state: dict[str, object] = {"points": [], "scale": 1.0}

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = float(state["scale"])
            state["points"].append((round(x / scale), round(y / scale)))

    cv2.setMouseCallback(window, mouse)
    for index, row in enumerate(pending, start=1):
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        scale = min(1.0, args.max_preview_width / image.shape[1])
        state.update({"points": [], "scale": scale})
        while True:
            canvas = image.copy()
            draw_polygon(canvas, row.get("goal_region_points_json", ""), (0, 140, 255), "PLANO DE ENTRADA")
            if row.get("ball_of_play_box_json"):
                x1, y1, x2, y2 = (round(float(value)) for value in json.loads(row["ball_of_play_box_json"]))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
                cv2.putText(canvas, "BOLA", (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 0), 2, cv2.LINE_AA)
            points = state["points"]
            if points:
                polygon = np.array(points, dtype="int32")
                cv2.polylines(canvas, [polygon], len(points) >= 3, (255, 160, 0), 3)
                for point in points:
                    cv2.circle(canvas, point, 5, (255, 160, 0), -1)
            cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 1200), 72), (0, 0, 0), -1)
            cv2.putText(canvas, f"{index}/{len(pending)} | {row['event_id']} | {row['match_id']} | {row['timestamp_hhmmss']}", (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Clique o contorno interno visivel da rede (>=4 pontos). G salvar | F copiar plano frontal | N nao visivel | U incerto | C limpar | Q sair", (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 120), 2, cv2.LINE_AA)
            rendered = cv2.resize(canvas, None, fx=scale, fy=scale) if scale != 1.0 else canvas
            cv2.imshow(window, rendered)
            key = cv2.waitKeyEx(60)
            if key in {-1, 255}:
                continue
            char = chr(key & 0xFF).lower()
            if char == "q" or key == 27:
                save_csv(output, rows, fieldnames)
                cv2.destroyAllWindows()
                print(f"Progresso salvo: {output}")
                return 0
            if char == "c":
                state["points"] = []
            elif char == "f":
                row["goal_volume_points_json"] = row["goal_region_points_json"]
                row["goal_volume_status"] = "COPIED_FRONT_PLANE"
                save_csv(output, rows, fieldnames)
                print(f"Salvo: {row['event_id']} -> COPIED_FRONT_PLANE")
                break
            elif char == "n":
                row["goal_volume_points_json"] = ""
                row["goal_volume_status"] = "NOT_VISIBLE"
                save_csv(output, rows, fieldnames)
                print(f"Salvo: {row['event_id']} -> NOT_VISIBLE")
                break
            elif char == "u":
                row["goal_volume_points_json"] = ""
                row["goal_volume_status"] = "UNCERTAIN"
                save_csv(output, rows, fieldnames)
                print(f"Salvo: {row['event_id']} -> UNCERTAIN")
                break
            elif char == "g":
                if len(points) < 4:
                    print("Clique pelo menos quatro pontos antes de salvar.")
                    continue
                row["goal_volume_points_json"] = json.dumps(points)
                row["goal_volume_status"] = "VALID"
                save_csv(output, rows, fieldnames)
                print(f"Salvo: {row['event_id']} -> VALID")
                break
    cv2.destroyAllWindows()
    print(f"Marcação concluída: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
