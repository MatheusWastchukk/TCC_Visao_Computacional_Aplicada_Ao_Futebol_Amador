"""Marca o instante decisivo de entrada da bola em lances positivos revisados."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


FIELDS = ["event_id", "match_id", "video_filename", "event_label", "event_timestamp_seconds", "event_time_hhmmss", "annotation_status"]


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Marca o frame em que a bola entra na região do gol.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    with args.events.open(newline="", encoding="utf-8") as file:
        events = [row for row in csv.DictReader(file) if row["event_label"] == "POSSIVEL_GOL"]
    existing: dict[str, dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["event_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(row["event_id"], {
        "event_id": row["event_id"], "match_id": row["match_id"], "video_filename": row["video_filename"],
        "event_label": row["event_label"], "event_timestamp_seconds": "", "event_time_hhmmss": "", "annotation_status": "pending",
    }) for row in events]
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print(f"Não há lances pendentes: {args.output}")
        return 0

    source_by_id = {row["event_id"]: row for row in events}
    window = "Marcação do momento decisivo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for event_index, row in enumerate(pending, start=1):
        source = source_by_id[row["event_id"]]
        capture = cv2.VideoCapture(str(args.videos_dir / row["video_filename"]))
        if not capture.isOpened():
            raise FileNotFoundError(args.videos_dir / row["video_filename"])
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            first_frame = max(0, round(float(source["start_seconds"]) * fps))
            last_frame = max(first_frame, round(float(source["end_seconds"]) * fps))
            current_frame, playing = first_frame, False
            while True:
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Não foi possível ler {row['video_filename']} no quadro {current_frame}.")
                seconds = current_frame / fps
                cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1100), 72), (0, 0, 0), -1)
                state = "RODANDO" if playing else "PAUSADO"
                cv2.putText(image, f"Lance {event_index}/{len(pending)} | {row['match_id']} | {format_time(seconds)} | {state}", (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Espaco: tocar/pausar | A/D: -/+ 1 s | J/L: -/+ 1 quadro | M: marcar entrada | R: reiniciar | Q: sair", (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 120), 2, cv2.LINE_AA)
                scale = min(1.0, args.max_preview_width / image.shape[1])
                canvas = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
                cv2.imshow(window, canvas)
                key = cv2.waitKeyEx(max(1, round(1000 / fps)) if playing else 60)
                if key in {-1, 255}:
                    if playing:
                        current_frame = min(last_frame, current_frame + 1)
                        if current_frame >= last_frame:
                            playing = False
                    continue
                char = chr(key & 0xFF).lower()
                if char == "q" or key == 27:
                    save_csv(args.output, rows)
                    cv2.destroyAllWindows()
                    print(f"Progresso salvo: {args.output}")
                    return 0
                if key == 32:
                    playing = not playing
                elif char == "r":
                    current_frame, playing = first_frame, False
                elif char == "a" or key == 2424832:
                    current_frame, playing = max(first_frame, current_frame - round(fps)), False
                elif char == "d" or key == 2555904:
                    current_frame, playing = min(last_frame, current_frame + round(fps)), False
                elif char == "j":
                    current_frame, playing = max(first_frame, current_frame - 1), False
                elif char == "l":
                    current_frame, playing = min(last_frame, current_frame + 1), False
                elif char == "m":
                    row["event_timestamp_seconds"] = f"{seconds:.3f}"
                    row["event_time_hhmmss"] = format_time(seconds)
                    row["annotation_status"] = "completed"
                    save_csv(args.output, rows)
                    print(f"Salvo: {row['event_id']} -> {format_time(seconds)}")
                    break
        finally:
            capture.release()
    cv2.destroyAllWindows()
    print(f"Marcação concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
