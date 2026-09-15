"""Anota clipes POSSIBLE_GOAL com reprodução, retomada segura e saída CSV."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2


DECISIONS = {"g": "GOAL", "n": "NO_GOAL", "u": "NOT_EVALUABLE"}


def format_time(seconds: float) -> str:
    """Formata segundos na forma minutos:segundos para a interface."""
    total = max(0, int(round(seconds)))
    minutes, seconds_part = divmod(total, 60)
    return f"{minutes:02d}:{seconds_part:02d}"


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Lê o manifesto CSV e valida sua chave estável de retomada."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {"segment_id", "clip_filename"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Manifesto sem colunas obrigatórias: {sorted(missing)}")
    if not rows:
        raise ValueError("Não há clipes no manifesto.")
    return rows, fields


def output_fields(source_fields: list[str]) -> list[str]:
    """Acrescenta os campos de anotação, preservando o manifesto de origem."""
    fields = list(source_fields)
    for field in ("human_label", "annotation_status", "notes"):
        if field not in fields:
            fields.append(field)
    return fields


def merge_existing(
    source_rows: list[dict[str, str]], fields: list[str], output: Path
) -> list[dict[str, str]]:
    """Preserva decisões já gravadas, sem modificar o CSV de entrada."""
    existing: dict[str, dict[str, str]] = {}
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as file:
            existing = {row["segment_id"]: row for row in csv.DictReader(file)}
    rows: list[dict[str, str]] = []
    for source in source_rows:
        previous = existing.get(source["segment_id"], {})
        row = {field: previous.get(field, source.get(field, "")) for field in fields}
        row["human_label"] = previous.get("human_label", "")
        row["annotation_status"] = previous.get("annotation_status", "pending")
        row["notes"] = previous.get("notes", "")
        rows.append(row)
    return rows


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Grava o progresso de forma atômica no diretório do anotador."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def default_directory() -> Path:
    """Usa a pasta do executável quando empacotado e o diretório atual no Python."""
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()


def draw_overlay(
    image,
    index: int,
    total: int,
    row: dict[str, str],
    seconds: float,
    duration: float,
    playing: bool,
) -> None:
    """Desenha instruções curtas sem esconder a maior parte do lance."""
    state = "RODANDO" if playing else "PAUSADO"
    decision = row.get("human_label") or "SEM RÓTULO"
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1250), 88), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"Clipe {index}/{total} | {row['segment_id']} | {format_time(seconds)}/{format_time(duration)} | {state} | {decision}",
        (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Espaço tocar/pausar | A/D -/+1s | J/L -/+5 quadros | G gol | N não gol | U não avaliável | W salvar | Q sair",
        (16, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 230, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "A decisão é salva automaticamente ao avançar. Retome depois abrindo este mesmo executável.",
        (16, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 240, 180), 1, cv2.LINE_AA,
    )


def annotate(rows: list[dict[str, str]], fields: list[str], clips_dir: Path, output: Path, max_width: int) -> int:
    """Exibe os clipes pendentes e registra a decisão humana por segmento."""
    pending = [row for row in rows if row.get("annotation_status") != "completed"]
    if not pending:
        print("Não há clipes pendentes.")
        return 0
    window = "Anotador de clipes POSSIBLE_GOAL"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, start=1):
        clip_path = clips_dir / row["clip_filename"]
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o clipe: {clip_path}")
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
                draw_overlay(image, index, len(pending), row, current_frame / fps, frame_count / fps, playing)
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
                    write_rows(output, fields, rows)
                    cv2.destroyAllWindows()
                    print(f"Progresso salvo: {output}")
                    return 0
                elif char == "a" or key == 2424832:
                    current_frame, playing = max(0, current_frame - round(fps)), False
                elif char == "d" or key == 2555904:
                    current_frame, playing = min(frame_count - 1, current_frame + round(fps)), False
                elif char == "j":
                    current_frame, playing = max(0, current_frame - 5), False
                elif char == "l":
                    current_frame, playing = min(frame_count - 1, current_frame + 5), False
                elif char in DECISIONS:
                    row["human_label"] = DECISIONS[char]
                    row["annotation_status"] = "completed"
                    write_rows(output, fields, rows)
                    print(f"Salvo: {row['segment_id']} -> {row['human_label']}")
                    break
                elif char == "w":
                    write_rows(output, fields, rows)
                    print(f"Progresso salvo: {output}")
        finally:
            capture.release()
    write_rows(output, fields, rows)
    cv2.destroyAllWindows()
    print(f"Anotação concluída: {output}")
    return 0


def main() -> int:
    base = default_directory()
    parser = argparse.ArgumentParser(description="Anota clipes POSSIBLE_GOAL com retomada segura.")
    parser.add_argument("--manifest", type=Path, default=base / "clips_manifest.csv")
    parser.add_argument("--clips-dir", type=Path, default=base)
    parser.add_argument("--output", type=Path, default=base / "clip_annotations.csv")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifesto não encontrado: {args.manifest}")
    if not args.clips_dir.is_dir():
        raise NotADirectoryError(f"Pasta de clipes não encontrada: {args.clips_dir}")
    source_rows, source_fields = read_rows(args.manifest)
    fields = output_fields(source_fields)
    rows = merge_existing(source_rows, fields, args.output)
    return annotate(rows, fields, args.clips_dir, args.output, args.max_preview_width)


if __name__ == "__main__":
    raise SystemExit(main())
