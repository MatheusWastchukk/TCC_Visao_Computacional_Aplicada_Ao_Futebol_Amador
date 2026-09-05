"""Revisa visualmente mudanças candidatas nos dígitos do placar."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


LABELS = {ord("g"): "SCORE_CHANGED", ord("n"): "NO_SCORE_CHANGE", ord("u"): "INCONCLUSIVE"}


def save(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load_crop(video_path: Path, timestamp: float, roi: list[int]) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Não foi possível ler {video_path} em {timestamp:.3f}s.")
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def main() -> int:
    parser = argparse.ArgumentParser(description="Revisa placar: G=placar mudou, N=não mudou, U=inconclusivo.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        source = list(csv.DictReader(file))
    existing = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["review_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(row["review_id"], row) for row in source]
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print("Não há revisões pendentes."); return 0
    window = "Revisão de mudança do placar"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, 1):
        roi = json.loads(row["scoreboard_roi_json"])
        before = load_crop(args.videos_dir / row["video_filename"], float(row["before_seconds"]), roi)
        after = load_crop(args.videos_dir / row["video_filename"], float(row["after_seconds"]), roi)
        scale = max(3.0, min(8.0, 500.0 / max(1, before.shape[1])))
        before = cv2.resize(before, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        after = cv2.resize(after, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        image = np.hstack((before, after))
        cv2.rectangle(image, (0, 0), (image.shape[1], 72), (0, 0, 0), -1)
        cv2.putText(image, f"{index}/{len(pending)} | {row['match_id']} | antes (-10s) à esquerda | depois (+10s) à direita | rank {row['rank_in_match']}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "G=placar mudou | N=nao mudou/ruido | U=inconclusivo | S=salvar | Q=sair", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 230, 140), 1, cv2.LINE_AA)
        cv2.imshow(window, image)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in LABELS:
                row["manual_label"] = LABELS[key]; row["annotation_status"] = "completed"; save(args.output, rows); break
            if key == ord("s"):
                save(args.output, rows); print(f"Progresso salvo: {args.output}")
            elif key in {ord("q"), 27}:
                save(args.output, rows); cv2.destroyAllWindows(); print(f"Progresso salvo: {args.output}"); return 0
    save(args.output, rows); cv2.destroyAllWindows(); print(f"Revisão concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
