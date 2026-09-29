"""Anota o fator visual predominante em uma auditoria curta de ranks de gol."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


FACTORS = {
    "l": "SINAL_LIMPO",
    "b": "BOLA_DIFICIL",
    "e": "ESTRUTURA_GOL_DIFICIL",
    "t": "TRAJETORIA_DIFICIL",
    "c": "CONTEXTO_CONFUSO",
    "u": "INCONCLUSIVO",
}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Lê o manifesto e garante os campos usados para abrir os clipes."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {"audit_id", "clip_filename", "match_id", "group"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Manifesto sem colunas obrigatórias: {sorted(missing)}")
    return rows, fields


def merge_existing(rows: list[dict[str, str]], fields: list[str], output: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Retoma escolhas anteriores sem modificar o manifesto original."""
    for field in ("primary_factor", "annotation_status"):
        if field not in fields:
            fields.append(field)
    existing: dict[str, dict[str, str]] = {}
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as file:
            existing = {row["audit_id"]: row for row in csv.DictReader(file)}
    merged = []
    for source in rows:
        previous = existing.get(source["audit_id"], {})
        row = {field: previous.get(field, source.get(field, "")) for field in fields}
        row["primary_factor"] = previous.get("primary_factor", "")
        row["annotation_status"] = previous.get("annotation_status", "pending")
        merged.append(row)
    return merged, fields


def save(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    """Persiste o progresso em arquivo temporário antes de substituir a saída."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def draw_overlay(image, index: int, total: int, row: dict[str, str], playing: bool) -> None:
    """Mostra o contraste de ranking e os seis atalhos de decisão."""
    state = "RODANDO" if playing else "PAUSADO"
    rank = row.get("rank_diverso") or "fora do pool"
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1250), 112), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"{index}/{total} | {row['match_id']} | {row['group']} | rank diverso: {rank} | {state}",
        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "L sinal limpo | B bola dificil | E estrutura do gol dificil | T trajetoria dificil",
        (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (120, 230, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "C contexto confuso | U inconclusivo | Espaco tocar/pausar | A/D -/+1s | J/F -/+5 quadros | Q sair",
        (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 230, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Escolha apenas o fator predominante; a decisao salva e avanca automaticamente.",
        (16, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 240, 180), 1, cv2.LINE_AA,
    )


def annotate(rows: list[dict[str, str]], fields: list[str], clips_root: Path, output: Path, max_width: int) -> int:
    """Exibe os itens pendentes e registra um fator predominante por clipe."""
    pending = [row for row in rows if row.get("annotation_status") != "completed"]
    if not pending:
        print("Não há itens pendentes na auditoria.")
        return 0
    window = "Auditoria de extremos de ranking"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, start=1):
        capture = cv2.VideoCapture(str(clips_root / row["clip_filename"]))
        if not capture.isOpened():
            raise FileNotFoundError(clips_root / row["clip_filename"])
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
            current_frame, playing = 0, False
            while True:
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ok, image = capture.read()
                if not ok or image is None:
                    current_frame = max(0, min(current_frame, frame_count - 1))
                    continue
                draw_overlay(image, index, len(pending), row, playing)
                if image.shape[1] > max_width:
                    scale = max_width / image.shape[1]
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                cv2.imshow(window, image)
                key = cv2.waitKeyEx(max(1, round(1000 / fps)) if playing else 0)
                if key in {-1, 255}:
                    if playing:
                        current_frame = min(frame_count - 1, current_frame + 1)
                        if current_frame == frame_count - 1:
                            playing = False
                    continue
                char = chr(key & 0xFF).lower()
                if key == 32 or char == " ":
                    playing = not playing
                elif char == "q" or key == 27:
                    save(output, rows, fields)
                    cv2.destroyAllWindows()
                    return 0
                elif char == "a" or key == 2424832:
                    current_frame, playing = max(0, current_frame - round(fps)), False
                elif char == "d" or key == 2555904:
                    current_frame, playing = min(frame_count - 1, current_frame + round(fps)), False
                elif char in FACTORS:
                    row["primary_factor"] = FACTORS[char]
                    row["annotation_status"] = "completed"
                    save(output, rows, fields)
                    break
                elif char == "j":
                    current_frame, playing = max(0, current_frame - 5), False
                elif char == "f":
                    current_frame, playing = min(frame_count - 1, current_frame + 5), False
        finally:
            capture.release()
    save(output, rows, fields)
    cv2.destroyAllWindows()
    print(f"Auditoria concluída: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Anota fatores visuais em extremos de ranking de gol.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    source_rows, fields = read_rows(args.manifest)
    rows, fields = merge_existing(source_rows, fields, args.output)
    return annotate(rows, fields, args.clips_root, args.output, args.max_preview_width)


if __name__ == "__main__":
    raise SystemExit(main())
