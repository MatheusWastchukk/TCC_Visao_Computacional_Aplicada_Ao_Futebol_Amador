"""Permite revisar relações espaciais entre o tracking da bola e uma goleira."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


LABELS = {
    "1": "RELACAO_PLAUSIVEL",
    "2": "ASSOCIACAO_INCORRETA",
    "3": "SEM_EVIDENCIA_UTIL",
    "4": "INCONCLUSIVO",
}


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Salva o estado atual sem alterar as colunas do manifesto."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Revisa overlays da relação espacial entre bola rastreada e goleira."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pending-only", action="store_true")
    args = parser.parse_args()

    output = args.output or args.input
    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("O manifesto de revisão está vazio.")
    for row in rows:
        row.setdefault("review_status", "")
        row.setdefault("review_label", "")
        row.setdefault("review_notes", "")

    pending = [row for row in rows if not args.pending_only or row["review_status"] != "complete"]
    if not pending:
        print(f"Não há sequências pendentes: {output}")
        return 0

    window = "Revisão bola e goleira"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    instructions = (
        "1 plausível | 2 associação incorreta | 3 sem evidência útil | "
        "4 inconclusivo | Q sair"
    )
    for index, row in enumerate(pending):
        image = cv2.imread(row["overlay_path"])
        if image is None:
            raise FileNotFoundError(row["overlay_path"])
        cv2.putText(
            image,
            f"{index + 1}/{len(pending)} | {instructions}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window, image)
        key = cv2.waitKeyEx(0)
        char = chr(key & 0xFF).lower() if key not in {-1, 255} else ""
        if char in {"q", "\x1b"} or key == 27:
            save_csv(output, rows)
            cv2.destroyAllWindows()
            print(f"Progresso salvo: {output}")
            return 0
        if char not in LABELS:
            continue
        row["review_label"] = LABELS[char]
        row["review_status"] = "complete"
        save_csv(output, rows)
        print(f"Salvo: {row['sequence_id']} ({index + 1}/{len(pending)})")

    cv2.destroyAllWindows()
    save_csv(output, rows)
    print(f"Revisão concluída: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
