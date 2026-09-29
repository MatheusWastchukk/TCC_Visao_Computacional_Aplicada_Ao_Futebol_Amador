"""Anota contexto visual em candidatos TRAIN sem alterar a etiqueta de gol."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


LABELS = {
    "o": "ACAO_OFENSIVA_MAS_NAO_GOL",
    "s": "SEM_ACAO_OFENSIVA",
    "i": "INCERTO",
    "b": "ERRO_DE_BOLA",
    "c": "CAMERA_OU_CONTEXTO",
    "t": "OUTRO",
}
DECISIONS = {
    "ACAO_OFENSIVA_MAS_NAO_GOL": "KEEP_FOR_REVIEW",
    "SEM_ACAO_OFENSIVA": "REJECT_NO_ACTION",
    "INCERTO": "KEEP_FOR_REVIEW",
    "ERRO_DE_BOLA": "REJECT_NO_ACTION",
    "CAMERA_OU_CONTEXTO": "REJECT_NO_ACTION",
    "OUTRO": "REJECT_NO_ACTION",
}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def merge_existing(manifest: Path, output: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows, fields = read_rows(manifest)
    identifier = "audit_id" if "audit_id" in fields else "context_id" if "context_id" in fields else ""
    if not identifier:
        raise ValueError("Manifesto sem audit_id ou context_id.")
    for field in ("context_label", "annotation_status", "notes"):
        if field not in fields:
            fields.append(field)
    existing = {row[identifier]: row for row in read_rows(output)[0]} if output.is_file() else {}
    merged = []
    for source in rows:
        previous = existing.get(source[identifier], {})
        row = {field: previous.get(field, source.get(field, "")) for field in fields}
        row["annotation_status"] = previous.get("annotation_status", source.get("annotation_status", "pending"))
        merged.append(row)
    return merged, fields


def save(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def draw(image, row: dict[str, str], index: int, total: int, playing: bool) -> None:
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1270), 110), (0, 0, 0), -1)
    state = "RODANDO" if playing else "PAUSADO"
    cv2.putText(image, f"{index}/{total} | {row['match_id']} | rank {row['temporal_rank']} | {state}", (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "O acao ofensiva | S sem acao | I incerto | B erro de bola | C camera/contexto | T acao fora da area", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Espaco tocar/pausar | A/D -/+1s | J/F -/+5 quadros | Q salvar e sair", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 240, 180), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Anota o contexto visual de falsos candidatos TRAIN.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    rows, fields = merge_existing(args.manifest, args.output)
    pending = [row for row in rows if row.get("annotation_status") != "completed"]
    if not pending:
        print("Não há itens pendentes na auditoria.")
        return 0
    window = "Auditoria de contexto ofensivo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, start=1):
        capture = cv2.VideoCapture(str(args.videos_dir / row["video_filename"]))
        if not capture.isOpened():
            raise FileNotFoundError(args.videos_dir / row["video_filename"])
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            start, end = float(row["delivery_start_seconds"]), float(row["delivery_end_seconds"])
            current, playing = start, False
            while True:
                capture.set(cv2.CAP_PROP_POS_MSEC, current * 1000.0)
                ok, image = capture.read()
                if not ok or image is None:
                    current, playing = start, False
                    continue
                draw(image, row, index, len(pending), playing)
                if image.shape[1] > args.max_preview_width:
                    scale = args.max_preview_width / image.shape[1]
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                cv2.imshow(window, image)
                key = cv2.waitKeyEx(max(1, round(1000 / fps)) if playing else 0)
                if key in {-1, 255} and playing:
                    current = min(end, current + 1.0 / fps)
                    if current >= end:
                        playing = False
                    continue
                char = chr(key & 0xFF).lower()
                if key == 32:
                    playing = not playing
                elif char == "q" or key == 27:
                    save(args.output, rows, fields)
                    cv2.destroyAllWindows()
                    return 0
                elif char == "a" or key == 2424832:
                    current, playing = max(start, current - 1.0), False
                elif char == "d" or key == 2555904:
                    current, playing = min(end, current + 1.0), False
                elif char == "j":
                    current, playing = max(start, current - 5.0 / fps), False
                elif char == "f":
                    current, playing = min(end, current + 5.0 / fps), False
                elif char in LABELS:
                    row["context_label"] = LABELS[char]
                    if "context_decision" in row:
                        row["context_decision"] = DECISIONS[LABELS[char]]
                    row["annotation_status"] = "completed"
                    save(args.output, rows, fields)
                    break
        finally:
            capture.release()
    save(args.output, rows, fields)
    cv2.destroyAllWindows()
    print(f"Auditoria concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
