"""Aplicação de teclado para anotar frames de reinício no centro."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


LABELS = {
    ord("r"): "REINICIO_CENTRO",
    ord("n"): "NAO_REINICIO_CENTRO",
    ord("u"): "INCONCLUSIVO",
}


def save(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frame(video_path: Path, timestamp: float) -> tuple[object, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Não foi possível ler {video_path} em {timestamp:.3f}s.")
        return image, fps
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Anota evidência visual de reinício no centro: R=sim, N=não, U=inconclusivo.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1600)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    existing: dict[str, dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["sample_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(row["sample_id"], row) for row in source_rows]
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print("Não há amostras pendentes.")
        return 0

    window = "Anotação de reinício no centro"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, 1):
        timestamp = float(row["sample_timestamp_seconds"])
        image, _ = load_frame(args.videos_dir / row["video_filename"], timestamp)
        header = f"{index}/{len(pending)} | {row['match_id']} | {row['sample_timestamp_hhmmss']}"
        help_text = "R=reinício no centro | N=não é reinício no centro | U=inconclusivo | S=salvar | Q=sair"
        cv2.rectangle(image, (0, 0), (min(image.shape[1], 1550), 72), (0, 0, 0), -1)
        cv2.putText(image, header, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, help_text, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (255, 230, 140), 1, cv2.LINE_AA)
        scale = min(1.0, args.max_preview_width / image.shape[1])
        preview = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
        cv2.imshow(window, preview)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in LABELS:
                row["restart_label"] = LABELS[key]
                row["annotation_status"] = "completed"
                save(args.output, rows)
                break
            if key == ord("s"):
                save(args.output, rows)
                print(f"Progresso salvo: {args.output}")
            elif key in {ord("q"), 27}:
                save(args.output, rows)
                cv2.destroyAllWindows()
                print(f"Progresso salvo: {args.output}")
                return 0
    save(args.output, rows)
    cv2.destroyAllWindows()
    print(f"Anotação concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
