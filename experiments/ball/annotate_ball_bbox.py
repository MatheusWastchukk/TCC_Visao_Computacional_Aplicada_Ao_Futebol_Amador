"""Anota a visibilidade e a caixa da bola em frames candidatos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


VISIBILITY_NAMES = {
    "v": "VISIBLE",
    "p": "PARTIAL",
    "n": "NOT_VISIBLE",
    "u": "UNCERTAIN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anota a bola nos frames candidatos.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1024)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field for field in rows[0] if field is not None]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") or "" for field in fields} for row in rows])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        all_rows = list(csv.DictReader(file))
    # Ao retomar, usa o progresso existente no arquivo de saída sem perder
    # candidatos que ainda não foram anotados.
    if args.output.exists():
        saved_rows = read_rows(args.output)
        saved_by_id = {row.get("sample_id"): row for row in saved_rows if row.get("sample_id")}
        all_rows = [saved_by_id.get(row.get("sample_id"), row) for row in all_rows]
    rows = (
        [row for row in all_rows if row.get("annotation_status") != "completed"]
        if args.pending_only
        else all_rows
    )
    index = 0
    bbox: tuple[int, int, int, int] | None = None
    visibility = "VISIBLE"
    scale = 1.0
    drawing = False
    start_point: tuple[int, int] | None = None

    window = "Anotador da bola | arraste a caixa"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal bbox, drawing, start_point
        original = (round(x / scale), round(y / scale))
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = original
        elif event == cv2.EVENT_MOUSEMOVE and drawing and start_point is not None:
            x1, y1 = start_point
            bbox = (min(x1, original[0]), min(y1, original[1]), max(x1, original[0]), max(y1, original[1]))
        elif event == cv2.EVENT_LBUTTONUP and start_point is not None:
            drawing = False
            x1, y1 = start_point
            bbox = (min(x1, original[0]), min(y1, original[1]), max(x1, original[0]), max(y1, original[1]))

    cv2.setMouseCallback(window, mouse_callback)
    try:
        while index < len(rows):
            row = rows[index]
            image = cv2.imread(str(Path(row["image_path"])))
            if image is None:
                raise FileNotFoundError(row["image_path"])
            scale = min(1.0, args.max_preview_width / image.shape[1])
            bbox = None
            visibility = "VISIBLE"
            drawing = False
            start_point = None
            while True:
                # Nunca desenhar diretamente sobre a imagem original: isso evita
                # acúmulo de caixas e borramento dos textos entre atualizações.
                preview = image.copy()
                if scale != 1.0:
                    preview = cv2.resize(preview, None, fx=scale, fy=scale)
                if bbox is not None:
                    x1, y1, x2, y2 = [round(value * scale) for value in bbox]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(preview, f"{index + 1}/{len(rows)} {row['sample_id']}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(preview, f"{row['timestamp_hhmmss']} | {visibility}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(preview, "V visivel | P parcial | N fora do frame | U incerto | arraste caixa | S salvar | R limpar | Q sair", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKeyEx(80)
                if key in {-1, 255}:
                    continue
                key_char = chr(key & 0xFF).lower()
                if key_char == "q" or key == 27:
                    write_rows(args.output, all_rows)
                    return 0
                if key_char in VISIBILITY_NAMES:
                    visibility = VISIBILITY_NAMES[key_char]
                    if visibility == "NOT_VISIBLE":
                        bbox = None
                elif key_char == "r":
                    bbox = None
                elif key_char == "s" or key in {13, 10}:
                    if visibility in {"VISIBLE", "PARTIAL"} and bbox is None:
                        print("Para VISIBLE/PARTIAL, arraste uma caixa em torno da bola.")
                        continue
                    row["ball_visibility"] = visibility
                    if bbox is None:
                        for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"):
                            row[field] = ""
                        if "ball_of_play_box_json" in row:
                            row["ball_of_play_box_json"] = ""
                    else:
                        row["ball_x1"], row["ball_y1"], row["ball_x2"], row["ball_y2"] = [str(value) for value in bbox]
                        if "ball_of_play_box_json" in row:
                            row["ball_of_play_box_json"] = json.dumps(list(bbox))
                    row["annotation_status"] = "completed"
                    row["annotator"] = row.get("annotator") or "manual"
                    write_rows(args.output, all_rows)
                    print(f"Salvo: {row['sample_id']} ({index + 1}/{len(rows)})")
                    index += 1
                    break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
