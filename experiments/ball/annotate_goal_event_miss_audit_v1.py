"""Anota a causa visual de gols TRAIN sem cobertura por candidatos."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


CATEGORY_BY_KEY = {
    "b": "BOLA_NAO_VISIVEL",
    "e": "ESTRUTURA_GOL_NAO_VISIVEL",
    "n": "BOLA_E_ESTRUTURA_NAO_VISIVEIS",
    "v": "BOLA_E_ESTRUTURA_VISIVEIS_SEM_CANDIDATO",
    "f": "GOL_FORA_DE_CAMERA",
    "t": "TIMESTAMP_CATALOGO_INCORRETO",
    "u": "INCONCLUSIVO",
}


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anota a causa de gols TRAIN sem cobertura por candidatos."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {
        "goal_catalog_id",
        "match_id",
        "goal_timestamp_seconds",
        "recommended_review_start_seconds",
        "annotation_status",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Fila sem colunas obrigatórias: {sorted(missing)}")
    for field in ("failure_category", "annotator_notes"):
        if field not in fields:
            fields.append(field)

    with args.split_manifest.open(newline="", encoding="utf-8") as file:
        split_rows = list(csv.DictReader(file))
    videos_by_match = {
        row["match_id"]: row["video_filename"]
        for row in split_rows
        if row.get("split") == "train"
    }
    unknown_matches = sorted({row["match_id"] for row in rows} - set(videos_by_match))
    if unknown_matches:
        raise AssertionError(
            f"A fila contém partidas fora de TRAIN ou ausentes do manifesto: {unknown_matches}"
        )

    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["goal_catalog_id"]: row for row in csv.DictReader(file)}
        rows = [
            {field: existing.get(row["goal_catalog_id"], row).get(field, "") for field in fields}
            for row in rows
        ]
    pending = [
        row for row in rows
        if not args.pending_only or row.get("annotation_status") != "completed"
    ]
    if not pending:
        print("Não há gols sem cobertura pendentes para auditoria.")
        return 0

    window = "Auditoria de gols sem cobertura"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, start=1):
        video_path = args.videos_dir / videos_by_match[row["match_id"]]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            target = float(row["goal_timestamp_seconds"])
            current_frame = min(
                round(float(row["recommended_review_start_seconds"]) * fps),
                max(0, frame_count - 1),
            )
            playing = False
            while True:
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ok, image = capture.read()
                if not ok or image is None:
                    current_frame = max(0, min(current_frame, frame_count - 1))
                    continue
                seconds = current_frame / fps
                header = (
                    f"{row['match_id']} | item {index}/{len(pending)} | "
                    f"agora {format_time(seconds)} | gol catalogado {format_time(target)}"
                )
                help_one = "ESPACO:play A/D:-/+5s J/L:-/+30s W:salvar Q:sair"
                help_two = "B:bola ausente E:trave ausente N:ambas ausentes V:ambas visiveis F:fora camera T:timestamp errado U:inconclusivo"
                cv2.putText(image, header, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(image, help_one, (16, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                cv2.putText(image, help_two, (16, image.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
                if image.shape[1] > args.max_preview_width:
                    scale = args.max_preview_width / image.shape[1]
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                cv2.imshow(window, image)
                key = cv2.waitKeyEx(max(1, round(1000 / fps)) if playing else 0)
                if key in {-1, 255}:
                    if playing:
                        current_frame = min(frame_count - 1, current_frame + 1)
                    continue
                char = chr(key & 0xFF).lower()
                if char == "q" or key == 27:
                    write_rows(args.output, fields, rows)
                    return 0
                if key == 32 or char == " ":
                    playing = not playing
                elif char == "a" or key == 2424832:
                    current_frame = max(0, current_frame - round(5 * fps))
                    playing = False
                elif char == "d" or key == 2555904:
                    current_frame = min(frame_count - 1, current_frame + round(5 * fps))
                    playing = False
                elif char == "j":
                    current_frame = max(0, current_frame - round(30 * fps))
                    playing = False
                elif char == "l":
                    current_frame = min(frame_count - 1, current_frame + round(30 * fps))
                    playing = False
                elif char in CATEGORY_BY_KEY:
                    row["failure_category"] = CATEGORY_BY_KEY[char]
                    row["annotation_status"] = "completed"
                    row["notes"] = (
                        "Causa visual da ausência de cobertura anotada manualmente no vídeo TRAIN."
                    )
                    write_rows(args.output, fields, rows)
                    break
                elif char == "w":
                    write_rows(args.output, fields, rows)
        finally:
            capture.release()
    write_rows(args.output, fields, rows)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
