"""Gera clipes de revisão do piloto de múltiplas hipóteses da bola."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def draw_box(image, values, color, label, thickness=2):
    """Desenha uma caixa xyxy e seu rótulo, quando disponível."""
    if not values:
        return
    x1, y1, x2, y2 = (round(float(value)) for value in values)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera clipes com alvo anotado, distratores e hipótese selecionada.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--include-matches", nargs="+", default=None, help="Gera clipes apenas para as partidas informadas.")
    args = parser.parse_args()

    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequence_rows = list(csv.DictReader(file))
    if args.include_matches:
        matches = set(args.include_matches)
        sequence_rows = [row for row in sequence_rows if row["match_id"] in matches]
        if not sequence_rows:
            parser.error("Nenhuma sequência corresponde às partidas informadas.")
    with args.results.open(newline="", encoding="utf-8") as file:
        result_rows = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in sequence_rows:
        grouped.setdefault(row["sequence_id"], []).append(row)
    args.output.mkdir(parents=True, exist_ok=True)

    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["sequence_frame"]))
        first = cv2.imread(rows[0]["image_path"])
        if first is None:
            raise FileNotFoundError(rows[0]["image_path"])
        height, width = first.shape[:2]
        output_path = args.output / f"{sequence_id}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Não foi possível criar: {output_path}")
        for row in rows:
            image = cv2.imread(row["image_path"])
            if image is None:
                raise FileNotFoundError(row["image_path"])
            result = result_rows[(sequence_id, row["sequence_frame"])]
            target = json.loads(row["ball_of_play_box_json"]) if row["ball_of_play_box_json"] else None
            distractors = json.loads(row["distractor_boxes_json"] or "[]")
            selected = json.loads(result["selected_box_json"]) if result.get("selected_box_json") else None
            draw_box(image, target, (0, 220, 0), "alvo", 2)
            for distractor in distractors:
                draw_box(image, distractor, (210, 0, 210), "distrator", 2)
            probability = result.get("selected_ball_probability", "")
            probability_label = f" | P(bola)={float(probability):.2f}" if probability else ""
            draw_box(image, selected, (0, 220, 255), f"hipótese {result['selected_candidate_rank']}{probability_label}", 3)
            status = "CORRETA" if result["identity_correct"] == "True" else "INCORRETA"
            cv2.rectangle(image, (8, 8), (460, 42), (0, 0, 0), -1)
            cv2.putText(image, f"{sequence_id} | frame {row['sequence_frame']} | {status}", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
        writer.release()
        print(f"Gerado: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
