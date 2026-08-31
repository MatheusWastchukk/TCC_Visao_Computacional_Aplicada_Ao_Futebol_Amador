"""Revisa as previsões e permite marcar manualmente a caixa da bola."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2


LABELS = {"1": "CORRETO", "2": "FALSO_NEGATIVO", "3": "FALSO_POSITIVO", "4": "INCONCLUSIVO"}
MANUAL_FIELDS = ("manual_ball_x1", "manual_ball_y1", "manual_ball_x2", "manual_ball_y2")


def parse_args() -> argparse.Namespace:
    base = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    default_csv = base / "ball_prediction_audit_v1.csv" if getattr(sys, "frozen", False) else base / "BALL-EXP-0001" / "audit_v1" / "ball_prediction_audit_v1.csv"
    parser = argparse.ArgumentParser(description="Revisa overlays e caixas da bola.")
    parser.add_argument("--input", type=Path, default=default_csv)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pending-only", action="store_true")
    return parser.parse_args()


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def existing_box(row: dict[str, str]) -> tuple[int, int, int, int] | None:
    try:
        box = tuple(int(float(row[field])) for field in MANUAL_FIELDS)
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def put_text(canvas, text: str, position: tuple[int, int], scale: float, color) -> None:
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main() -> int:
    args = parse_args()
    output = args.output or args.input
    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in MANUAL_FIELDS:
            row.setdefault(field, "")

    pending = [row for row in rows if not args.pending_only or row.get("review_status") != "complete"]
    if not pending:
        print("Não há amostras pendentes.")
        return 0

    window = "Auditoria da bola"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state: dict[str, object] = {"dragging": False, "start": None, "box": None, "base": None, "row": None}

    def render() -> None:
        base = state["base"]
        row = state["row"]
        if base is None or row is None:
            return
        canvas = base.copy()
        box = state["box"]
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 3)
            put_text(canvas, "CAIXA MANUAL", (x1, max(25, y1 - 8)), 0.8, (255, 0, 255))
        index = pending.index(row) + 1
        put_text(canvas, f"{index}/{len(pending)}  {row['sample_id']}", (20, 35), 0.85, (255, 255, 0))
        put_text(canvas, f"preliminar: {row['preliminary_status']} | GT: {row['ball_visibility']}", (20, 70), 0.7, (255, 255, 255))
        put_text(canvas, "Arraste: caixa manual | R: limpar | 1 correto | 2 FN | 3 FP | 4 inconclusivo", (20, 105), 0.58, (0, 220, 255))
        put_text(canvas, "S: salvar caixa e continuar | Q: sair salvando", (20, 135), 0.58, (0, 220, 255))
        state["canvas"] = canvas
        cv2.imshow(window, canvas)

    def mouse_callback(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            start = state["start"]
            if start:
                state["box"] = (min(start[0], x), min(start[1], y), max(start[0], x), max(start[1], y))
                render()
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            start = state["start"]
            if start:
                box = (min(start[0], x), min(start[1], y), max(start[0], x), max(start[1], y))
                state["box"] = box if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 else None
                render()

    cv2.setMouseCallback(window, mouse_callback)
    index = 0
    try:
        while index < len(pending):
            row = pending[index]
            image_path = Path(row["overlay_path"])
            if not image_path.exists() and getattr(sys, "frozen", False):
                image_path = Path(sys.executable).resolve().parents[4] / image_path
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(str(image_path))
            state["base"] = image
            state["row"] = row
            state["box"] = existing_box(row)
            render()
            key = cv2.waitKeyEx(0)
            char = chr(key & 0xFF).lower() if key not in {-1, 255} else ""
            if char in {"q", "\x1b"} or key == 27:
                save_csv(output, rows)
                return 0
            if char == "r":
                state["box"] = None
                for field in MANUAL_FIELDS:
                    row[field] = ""
                render()
                continue
            if char in {"s", *LABELS}:
                box = state["box"]
                if box:
                    for field, value in zip(MANUAL_FIELDS, box):
                        row[field] = str(value)
                if char in LABELS:
                    row["review_label"] = LABELS[char]
                    row["review_status"] = "complete"
                save_csv(output, rows)
                action = "Salvo" if char in LABELS else "Caixa salva"
                print(f"{action}: {row['sample_id']} ({index + 1}/{len(pending)})")
                index += 1
    finally:
        cv2.destroyAllWindows()
    save_csv(output, rows)
    print(f"Revisão concluída: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
