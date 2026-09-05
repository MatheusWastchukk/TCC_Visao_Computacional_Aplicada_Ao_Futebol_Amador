"""Aplicação para revisar visualmente mudança de placar antes e depois do evento."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def save(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def frame(path: Path, timestamp: float) -> np.ndarray:
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
        raise RuntimeError(f"Não foi possível ler {path} em {timestamp:.3f}s.")
    return image


def draw_roi(image: np.ndarray, roi_json: str) -> np.ndarray:
    output = image.copy()
    if roi_json:
        x, y, width, height = json.loads(roi_json)
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 220, 255), 2)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara placar antes/depois: O=delimitar, C=alterou, N=não alterou, X=sem placar, U=inconclusivo.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--missing-roi-only", action="store_true", help="Reabre somente contextos com placar disponível, mas sem região delimitada.")
    parser.add_argument("--max-preview-width", type=int, default=1600)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        original = list(csv.DictReader(file))
    existing = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["scoreboard_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(row["scoreboard_id"], row) for row in original]
    if args.missing_roi_only:
        pending = [row for row in rows if row["scoreboard_available"] == "True" and not row["scoreboard_roi_json"]]
    else:
        pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print("Não há contextos pendentes."); return 0
    window = "Evidência de placar"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for index, row in enumerate(pending, 1):
        before = frame(args.videos_dir / row["video_filename"], float(row["before_timestamp_seconds"]))
        after = frame(args.videos_dir / row["video_filename"], float(row["after_timestamp_seconds"]))
        while True:
            left, right = draw_roi(before, row["scoreboard_roi_json"]), draw_roi(after, row["scoreboard_roi_json"])
            height = min(left.shape[0], right.shape[0]); left = left[:height]; right = right[:height]
            image = np.hstack((left, right))
            cv2.rectangle(image, (0, 0), (min(image.shape[1], 1550), 73), (0, 0, 0), -1)
            decision = row["score_changed"] or "sem decisão"
            header = f"{index}/{len(pending)} | {row['match_id']} | antes {row['before_hhmmss']} | depois {row['after_hhmmss']} | decisão: {decision}"
            help_text = "O=delimitar placar | V=salvar região mantendo decisão | C=alterou | N=não alterou | X=sem placar | U=inconclusivo | S=salvar | Q=sair"
            cv2.putText(image, header, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, help_text, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 230, 140), 1, cv2.LINE_AA)
            scale = min(1.0, args.max_preview_width / image.shape[1])
            shown = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
            cv2.imshow(window, shown)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("o"):
                roi = cv2.selectROI("Delimite o placar", after, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow("Delimite o placar")
                if roi[2] > 0 and roi[3] > 0:
                    row["scoreboard_available"] = "True"; row["scoreboard_roi_json"] = json.dumps(list(map(int, roi)))
            elif key == ord("v") and row["scoreboard_roi_json"]:
                row["annotation_status"] = "completed"; save(args.output, rows); break
            elif key in {ord("c"), ord("n"), ord("x"), ord("u")}:
                if key == ord("c"):
                    row["scoreboard_available"] = "True"; row["score_changed"] = "True"
                elif key == ord("n"):
                    row["scoreboard_available"] = "True"; row["score_changed"] = "False"
                elif key == ord("x"):
                    row["scoreboard_available"] = "False"; row["score_changed"] = ""
                else:
                    row["scoreboard_available"] = ""; row["score_changed"] = ""
                row["annotation_status"] = "completed"; save(args.output, rows); break
            elif key == ord("s"):
                save(args.output, rows); print(f"Progresso salvo: {args.output}")
            elif key in {ord("q"), 27}:
                save(args.output, rows); cv2.destroyAllWindows(); print(f"Progresso salvo: {args.output}"); return 0
    save(args.output, rows); cv2.destroyAllWindows(); print(f"Anotação concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
