"""Marca o polígono interno da goleira no frame decisivo de cada evento positivo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


EXTRA_FIELDS = ["goal_region_points_json", "goal_region_status"]


def save_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Marca quatro pontos da região interna da goleira no momento decisivo.")
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
    candidates = [row for row in rows if row.get("is_event_frame") == "True"]
    pending = [row for row in candidates if not args.pending_only or not row.get("goal_region_status")]
    if not pending:
        print(f"Não há regiões pendentes: {output}")
        return 0

    window = "Marcação da região interna do gol"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state: dict[str, object] = {"points": [], "scale": 1.0}

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 4:
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
            if row.get("ball_of_play_box_json"):
                x1, y1, x2, y2 = (round(float(value)) for value in json.loads(row["ball_of_play_box_json"]))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
                cv2.putText(canvas, "BOLA", (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 0), 2, cv2.LINE_AA)
            points = state["points"]
            for point_index, point in enumerate(points, start=1):
                cv2.circle(canvas, point, 6, (255, 160, 0), -1)
                cv2.putText(canvas, str(point_index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2, cv2.LINE_AA)
            if len(points) >= 2:
                cv2.polylines(canvas, [np.array(points, dtype="int32")], False, (255, 160, 0), 2)
            if len(points) == 4:
                cv2.polylines(canvas, [np.array(points, dtype="int32")], True, (255, 160, 0), 3)
            cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 1100), 72), (0, 0, 0), -1)
            cv2.putText(canvas, f"{index}/{len(pending)} | {row['event_id']} | {row['match_id']} | {row['timestamp_hhmmss']}", (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Clique 4 pontos: sup-esq, sup-dir, inf-dir, inf-esq | G salvar | N gol nao visivel | U incerto | C limpar | Q sair", (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (255, 220, 120), 2, cv2.LINE_AA)
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
            elif char == "n":
                row["goal_region_points_json"] = ""
                row["goal_region_status"] = "NOT_VISIBLE"
                save_csv(output, rows, fieldnames)
                print(f"Salvo: {row['event_id']} -> NOT_VISIBLE")
                break
            elif char == "u":
                row["goal_region_points_json"] = ""
                row["goal_region_status"] = "UNCERTAIN"
                save_csv(output, rows, fieldnames)
                print(f"Salvo: {row['event_id']} -> UNCERTAIN")
                break
            elif char == "g":
                if len(state["points"]) != 4:
                    print("Clique exatamente quatro pontos antes de salvar.")
                    continue
                row["goal_region_points_json"] = json.dumps(state["points"])
                row["goal_region_status"] = "VALID"
                save_csv(output, rows, fieldnames)
                print(f"Salvo: {row['event_id']} -> VALID")
                break
    cv2.destroyAllWindows()
    print(f"Marcação concluída: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
