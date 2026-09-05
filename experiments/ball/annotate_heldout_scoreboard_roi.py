"""Delimita os dígitos do placar em partidas fechadas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def save(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load_frame(path: Path, timestamp: float) -> object:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Não foi possível ler {path}.")
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Delimita dígitos do placar em partidas fechadas: O=selecionar, V=salvar, Q=sair.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=300.0)
    parser.add_argument("--pending-only", action="store_true")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        source = list(csv.DictReader(file))
    existing = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["match_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(row["match_id"], row) for row in source]
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    window = "Delimitação do placar fechado"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, 1):
        image = load_frame(args.videos_dir / row["video_filename"], args.sample_seconds)
        while True:
            shown = image.copy()
            if row["scoreboard_roi_json"]:
                x, y, width, height = json.loads(row["scoreboard_roi_json"])
                cv2.rectangle(shown, (x, y), (x + width, y + height), (0, 220, 255), 3)
            cv2.rectangle(shown, (0, 0), (min(shown.shape[1], 1500), 72), (0, 0, 0), -1)
            cv2.putText(shown, f"{index}/{len(pending)} | {row['match_id']} | gols esperados: {row['expected_goals']}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.putText(shown, "O=delimitar somente os dígitos do placar; exclua relógio/logos | V=salvar | Q=sair", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 230, 140), 1)
            cv2.imshow(window, shown)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("o"):
                roi = cv2.selectROI("Delimite os dígitos do placar", image, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow("Delimite os dígitos do placar")
                if roi[2] > 0 and roi[3] > 0:
                    row["scoreboard_roi_json"] = json.dumps(list(map(int, roi)))
            elif key == ord("v") and row["scoreboard_roi_json"]:
                row["annotation_status"] = "completed"; save(args.output, rows); break
            elif key in {ord("q"), 27}:
                save(args.output, rows); cv2.destroyAllWindows(); print(f"Progresso salvo: {args.output}"); return 0
    save(args.output, rows); cv2.destroyAllWindows(); print(f"Delimitação concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
