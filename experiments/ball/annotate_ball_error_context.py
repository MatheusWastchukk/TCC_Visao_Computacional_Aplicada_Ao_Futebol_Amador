"""Classifica o contexto principal dos falsos negativos da bola."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


CONTEXTS = {
    "1": "MAOS_GOLEIRO",
    "2": "COBRANCA_LATERAL",
    "3": "OCULTA_JOGADOR",
    "4": "MUITO_PEQUENA",
    "5": "BORRADA_BAIXA_QUALIDADE",
    "6": "FORA_ENQUADRAMENTO",
    "7": "OUTRO",
    "8": "INCONCLUSIVO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classifica o contexto dos falsos negativos.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pending-only", action="store_true")
    return parser.parse_args()


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_manual_box(image, row: dict[str, str]) -> None:
    try:
        box = tuple(int(float(row[f"manual_ball_{axis}"])) for axis in ("x1", "y1", "x2", "y2"))
    except (TypeError, ValueError):
        return
    x1, y1, x2, y2 = box
    if x2 > x1 and y2 > y1:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(image, "CAIXA MANUAL", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)


def main() -> int:
    args = parse_args()
    output = args.output or args.input
    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row.setdefault("error_context", "")
        row.setdefault("context_status", "pending")

    candidates = [
        row for row in rows
        if row.get("review_label") == "FALSO_NEGATIVO"
        and (not args.pending_only or row.get("context_status") != "complete")
    ]
    if not candidates:
        print("Não há falsos negativos pendentes para classificar.")
        return 0

    window = "Contexto do erro da bola"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    index = 0
    try:
        while index < len(candidates):
            row = candidates[index]
            image = cv2.imread(str(Path(row["overlay_path"])))
            if image is None:
                raise FileNotFoundError(row["overlay_path"])
            draw_manual_box(image, row)
            cv2.putText(image, f"{index + 1}/{len(candidates)}  {row['sample_id']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
            cv2.putText(image, f"GT: {row['ball_visibility']} | contexto atual: {row['error_context'] or 'pendente'}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "1 maos goleiro | 2 lateral | 3 ocultacao | 4 pequena", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(image, "5 borrada/qualidade | 6 fora quadro | 7 outro | 8 inconclusivo | Q sair", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.imshow(window, image)
            key = cv2.waitKeyEx(0)
            char = chr(key & 0xFF).lower() if key not in {-1, 255} else ""
            if char in {"q", "\x1b"} or key == 27:
                save_csv(output, rows)
                return 0
            if char in CONTEXTS:
                row["error_context"] = CONTEXTS[char]
                row["context_status"] = "complete"
                save_csv(output, rows)
                print(f"Salvo: {row['sample_id']} ({index + 1}/{len(candidates)})")
                index += 1
    finally:
        cv2.destroyAllWindows()
    save_csv(output, rows)
    print(f"Classificação concluída: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
