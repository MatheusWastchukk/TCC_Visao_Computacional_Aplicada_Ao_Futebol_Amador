"""Anota qual candidata representa a bola da jogada em frames ambíguos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Persiste o progresso imediatamente após cada anotação."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contains(box: list[float], point: tuple[float, float]) -> bool:
    """Testa se um clique está dentro de uma caixa candidata."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identifica a bola da jogada entre múltiplas candidatas detectadas."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pending-only", action="store_true")
    args = parser.parse_args()
    output = args.output or args.input
    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("O manifesto de identidade está vazio.")
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "complete"]
    if not pending:
        print(f"Não há frames pendentes: {output}")
        return 0

    window = "Identidade da bola"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending):
        image = cv2.imread(row["frame_path"])
        if image is None:
            raise FileNotFoundError(row["frame_path"])
        boxes = json.loads(row["candidate_boxes_json"])
        confidences = json.loads(row["candidate_confidences_json"])
        previous = [float(value) for value in json.loads(row["previous_track_box_json"])]
        cv2.rectangle(image, (int(previous[0]), int(previous[1])), (int(previous[2]), int(previous[3])), (0, 0, 255), 3)
        cv2.putText(image, "TRACK ANTERIOR", (int(previous[0]), max(25, int(previous[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        for candidate_index, (box, confidence) in enumerate(zip(boxes, confidences), start=1):
            x1, y1, x2, y2 = (int(value) for value in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(image, f"C{candidate_index} {confidence:.2f}", (x1, max(45, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        message = (
            f"{index + 1}/{len(pending)} | clique ou tecla 1-9 na candidata correta | "
            "N nao detectada | I inconclusivo | Q sair"
        )
        cv2.putText(image, message, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        selected: list[int] = []

        def on_mouse(event, x, y, _flags, _param) -> None:
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            window_x, window_y, width, height = cv2.getWindowImageRect(window)
            if width <= 0 or height <= 0:
                return
            image_x = x * image.shape[1] / width
            image_y = y * image.shape[0] / height
            matches = [candidate_index for candidate_index, box in enumerate(boxes) if contains(box, (image_x, image_y))]
            if matches:
                selected[:] = [matches[0]]

        cv2.setMouseCallback(window, on_mouse)
        while not selected:
            cv2.imshow(window, image)
            key = cv2.waitKeyEx(50)
            char = chr(key & 0xFF).lower() if key not in {-1, 255} else ""
            if char in {"q", "\x1b"} or key == 27:
                save_csv(output, rows)
                cv2.destroyAllWindows()
                print(f"Progresso salvo: {output}")
                return 0
            if char == "n":
                row["target_candidate_id"] = ""
                row["target_visibility"] = "NAO_DETECTADA"
                row["annotation_status"] = "complete"
                break
            if char == "i":
                row["target_candidate_id"] = ""
                row["target_visibility"] = "INCONCLUSIVO"
                row["annotation_status"] = "complete"
                break
            if char.isdigit() and char != "0" and int(char) <= len(boxes):
                selected[:] = [int(char) - 1]
        if selected:
            row["target_candidate_id"] = f"C{selected[0] + 1}"
            row["target_visibility"] = "DETECTADA"
            row["annotation_status"] = "complete"
        save_csv(output, rows)
        print(f"Salvo: {row['ambiguous_id']} ({index + 1}/{len(pending)})")
    cv2.destroyAllWindows()
    print(f"Anotação concluída: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
