"""Permite revisar por teclas o efeito do score visual no multi-tracker."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


FIELDNAMES = [
    "sequence_id", "sequence_frame", "sample_id", "match_id", "image_path",
    "baseline_selected_rank", "baseline_identity_correct", "baseline_wrong_ball_switch",
    "score_selected_rank", "score_ball_probability", "score_identity_correct", "score_wrong_ball_switch",
    "manual_verdict", "annotation_status",
]


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def draw_box(image, box_json: str, color: tuple[int, int, int], label: str) -> None:
    if not box_json:
        return
    x1, y1, x2, y2 = (round(float(value)) for value in json.loads(box_json))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(image, label, (x1, max(25, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def render_panel(image, sequence_row: dict[str, str], result: dict[str, str], title: str, score_panel: bool) -> object:
    canvas = image.copy()
    target = sequence_row.get("ball_of_play_box_json", "")
    draw_box(canvas, target, (0, 220, 0), "ALVO")
    for distractor in json.loads(sequence_row.get("distractor_boxes_json") or "[]"):
        draw_box(canvas, json.dumps(distractor), (210, 0, 210), "DISTRATOR")
    suffix = ""
    if score_panel and result.get("selected_ball_probability"):
        suffix = f" | P(BOLA)={float(result['selected_ball_probability']):.2f}"
    draw_box(canvas, result.get("selected_box_json", ""), (0, 220, 255), f"HIPOTESE {result.get('selected_candidate_rank', '0')}{suffix}")
    status = "CORRETA" if result.get("identity_correct") == "True" else "INCORRETA"
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 720), 42), (0, 0, 0), -1)
    cv2.putText(canvas, f"{title} | {status}", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Revisa por teclas baseline versus score visual do multi-tracker.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-matches", nargs="+", default=None)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-panel-width", type=int, default=900)
    args = parser.parse_args()

    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequences = [row for row in csv.DictReader(file) if row.get("ball_of_play_box_json")]
    if args.include_matches:
        matches = set(args.include_matches)
        sequences = [row for row in sequences if row["match_id"] in matches]
    with args.baseline.open(newline="", encoding="utf-8") as file:
        baseline = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    with args.score.open(newline="", encoding="utf-8") as file:
        score = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}

    existing: dict[tuple[str, str], dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    rows: list[dict[str, str]] = []
    for sequence in sequences:
        key = (sequence["sequence_id"], sequence["sequence_frame"])
        if key not in baseline or key not in score:
            raise KeyError(f"Resultado ausente para {key[0]} quadro {key[1]}.")
        base, scored = baseline[key], score[key]
        row = existing.get(key, {
            "sequence_id": sequence["sequence_id"], "sequence_frame": sequence["sequence_frame"],
            "sample_id": sequence["sample_id"], "match_id": sequence["match_id"], "image_path": sequence["image_path"],
            "baseline_selected_rank": base["selected_candidate_rank"], "baseline_identity_correct": base["identity_correct"],
            "baseline_wrong_ball_switch": base["wrong_ball_switch"], "score_selected_rank": scored["selected_candidate_rank"],
            "score_ball_probability": scored.get("selected_ball_probability", ""), "score_identity_correct": scored["identity_correct"],
            "score_wrong_ball_switch": scored["wrong_ball_switch"], "manual_verdict": "", "annotation_status": "pending",
        })
        rows.append({field: row.get(field, "") for field in FIELDNAMES})
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print(f"Não há frames pendentes: {args.output}")
        return 0

    lookup = {(row["sequence_id"], row["sequence_frame"]): row for row in sequences}
    window = "Revisão: baseline x score visual"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    labels = {"m": "MELHOROU", "i": "IGUAL", "p": "PIOROU", "u": "INCONCLUSIVO"}
    for index, row in enumerate(pending, start=1):
        sequence = lookup[(row["sequence_id"], row["sequence_frame"])]
        image = cv2.imread(sequence["image_path"])
        if image is None:
            raise FileNotFoundError(sequence["image_path"])
        base = baseline[(row["sequence_id"], row["sequence_frame"])]
        scored = score[(row["sequence_id"], row["sequence_frame"])]
        left = render_panel(image, sequence, base, "BASELINE", False)
        right = render_panel(image, sequence, scored, "COM SCORE", True)
        scale = min(1.0, args.max_panel_width / image.shape[1])
        if scale != 1.0:
            left, right = cv2.resize(left, None, fx=scale, fy=scale), cv2.resize(right, None, fx=scale, fy=scale)
        canvas = cv2.hconcat([left, right])
        cv2.rectangle(canvas, (0, canvas.shape[0] - 76), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
        cv2.putText(canvas, f"{index}/{len(pending)} | {row['match_id']} | {row['sequence_id']} | quadro {row['sequence_frame']}", (15, canvas.shape[0] - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "M melhorou | I igual | P piorou | U inconclusivo | Q salva e sai", (15, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 2, cv2.LINE_AA)
        while True:
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
            if char in labels:
                row["manual_verdict"] = labels[char]
                row["annotation_status"] = "completed"
                save_csv(args.output, rows)
                print(f"Salvo: {index}/{len(pending)} -> {labels[char]}")
                break
    cv2.destroyAllWindows()
    print(f"Revisão concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
