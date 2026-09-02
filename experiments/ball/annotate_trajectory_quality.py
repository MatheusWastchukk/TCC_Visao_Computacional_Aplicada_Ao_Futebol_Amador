"""Revisa por clipe se a trajetória selecionada é útil para análise posterior."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


FIELDS = ["sequence_id", "match_id", "frames", "manual_quality", "annotation_status"]


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def draw_box(image, box_json: str, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    if not box_json:
        return
    x1, y1, x2, y2 = (round(float(value)) for value in json.loads(box_json))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classifica a utilidade temporal de trajetórias anotadas.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    with args.sequences.open(newline="", encoding="utf-8") as file:
        frames = [row for row in csv.DictReader(file) if row.get("ball_of_play_box_json")]
    with args.results.open(newline="", encoding="utf-8") as file:
        results = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in frames:
        grouped.setdefault(row["sequence_id"], []).append(row)
    existing: dict[str, dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["sequence_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(sequence_id, {
        "sequence_id": sequence_id, "match_id": sequence_frames[0]["match_id"], "frames": str(len(sequence_frames)),
        "manual_quality": "", "annotation_status": "pending",
    }) for sequence_id, sequence_frames in sorted(grouped.items())]
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print(f"Não há clipes pendentes: {args.output}")
        return 0

    window = "Revisão de qualidade da trajetória"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    labels = {"u": "UTIL", "p": "PARCIAL", "i": "INUTIL"}
    for clip_index, review in enumerate(pending, start=1):
        sequence_frames = sorted(grouped[review["sequence_id"]], key=lambda row: int(row["sequence_frame"]))
        frame_index = 0
        while True:
            row = sequence_frames[frame_index]
            result = results[(row["sequence_id"], row["sequence_frame"])]
            image = cv2.imread(row["image_path"])
            if image is None:
                raise FileNotFoundError(row["image_path"])
            draw_box(image, row["ball_of_play_box_json"], (0, 220, 0), "ALVO")
            for distractor in json.loads(row.get("distractor_boxes_json") or "[]"):
                draw_box(image, json.dumps(distractor), (210, 0, 210), "DISTRATOR")
            draw_box(image, result.get("selected_box_json", ""), (0, 220, 255), "TRACKER", 3)
            status = "CORRETA" if result["identity_correct"] == "True" else "INCORRETA"
            cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 850), 70), (0, 0, 0), -1)
            cv2.putText(image, f"Clipe {clip_index}/{len(pending)} | {review['sequence_id']} | quadro {frame_index + 1}/{len(sequence_frames)} | {status}", (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "A/D ou setas: navegar | U util | P parcial | I inutil | Q salva e sai", (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 120), 2, cv2.LINE_AA)
            scale = min(1.0, args.max_preview_width / image.shape[1])
            canvas = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(60)
            if key in {-1, 255}:
                continue
            char = chr(key & 0xFF).lower()
            if char == "q" or key == 27:
                save_csv(args.output, rows)
                cv2.destroyAllWindows()
                print(f"Progresso salvo: {args.output}")
                return 0
            if char == "a" or key == 2424832:
                frame_index = max(0, frame_index - 1)
                continue
            if char == "d" or key == 2555904:
                frame_index = min(len(sequence_frames) - 1, frame_index + 1)
                continue
            if char in labels:
                review["manual_quality"] = labels[char]
                review["annotation_status"] = "completed"
                save_csv(args.output, rows)
                print(f"Salvo: {review['sequence_id']} -> {labels[char]}")
                break
    cv2.destroyAllWindows()
    print(f"Revisão concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
