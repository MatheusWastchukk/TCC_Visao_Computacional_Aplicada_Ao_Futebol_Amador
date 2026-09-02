"""Aplicação para classificar candidatas falsas do detector da bola."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


LABELS = {
    ord("b"): "BOLA_JOGO",
    ord("d"): "BOLA_DISTRATORA",
    ord("p"): "PE_PERNA",
    ord("h"): "CABECA_TRONCO",
    ord("o"): "OUTRO",
    ord("i"): "INCONCLUSIVO",
}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classifica candidatas: B bola jogo, D distratora, P pé, H cabeça, O outro, I inconclusivo.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            saved = {row["candidate_id"]: row for row in csv.DictReader(file)}
        rows = [saved.get(row["candidate_id"], row) for row in rows]
    pending = [index for index, row in enumerate(rows) if row["annotation_status"] != "completed"] if args.pending_only else list(range(len(rows)))
    if not pending:
        print("Não há candidatas pendentes.")
        return 0
    current = 0
    window = "Classificador de candidatas da bola"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while current < len(pending):
        row_index = pending[current]
        row = rows[row_index]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        x1, y1, x2, y2 = (round(value) for value in json.loads(row["box_json"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 3)
        header = f"{current + 1}/{len(pending)} | {row['candidate_id']} | conf={float(row['confidence']):.3f}"
        help_text = "B=jogo D=distratora P=pe H=cabeca O=outro I=inconclusivo | S=salvar Q=sair"
        cv2.rectangle(image, (0, 0), (min(image.shape[1], 1250), 74), (0, 0, 0), -1)
        cv2.putText(image, header, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, help_text, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, image)
        key = cv2.waitKey(0) & 0xFF
        if key in LABELS:
            row["label"] = LABELS[key]
            row["annotation_status"] = "completed"
            row["label_source"] = "revisao_manual"
            write_rows(args.output, rows)
            current += 1
        elif key == ord("s"):
            write_rows(args.output, rows)
            print(f"Progresso salvo: {args.output}")
        elif key in {ord("q"), 27}:
            break
    write_rows(args.output, rows)
    cv2.destroyAllWindows()
    print(f"Progresso salvo: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
