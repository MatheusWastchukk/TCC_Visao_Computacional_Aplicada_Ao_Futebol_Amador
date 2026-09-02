"""Anota a bola da jogada e bolas distratoras em sequências temporais."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Anota a identidade temporal da bola em frames de treino.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    output = args.output or args.input
    with args.input.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print(f"Não há frames pendentes: {output}")
        return 0

    window = "Identidade temporal da bola"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state: dict[str, object] = {"drawing": False, "start": None, "draft": None, "scale": 1.0}

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        scale = float(state["scale"])
        point = (round(x / scale), round(y / scale))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = point
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"] and state["start"]:
            start = state["start"]
            state["draft"] = (min(start[0], point[0]), min(start[1], point[1]), max(start[0], point[0]), max(start[1], point[1]))
        elif event == cv2.EVENT_LBUTTONUP and state["start"]:
            state["drawing"] = False
            start = state["start"]
            state["draft"] = (min(start[0], point[0]), min(start[1], point[1]), max(start[0], point[0]), max(start[1], point[1]))

    cv2.setMouseCallback(window, mouse)
    for index, row in enumerate(pending):
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        scale = min(1.0, args.max_preview_width / image.shape[1])
        state.update({"drawing": False, "start": None, "draft": None, "scale": scale})
        target: tuple[int, int, int, int] | None = None
        distractors: list[tuple[int, int, int, int]] = []
        visibility = row.get("ball_visibility") or "VISIBLE"
        while True:
            canvas = image.copy()
            if target:
                cv2.rectangle(canvas, target[:2], target[2:], (0, 255, 0), 3)
                cv2.putText(canvas, "BOLA DA JOGADA", (target[0], max(25, target[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for box in distractors:
                cv2.rectangle(canvas, box[:2], box[2:], (0, 0, 255), 3)
                cv2.putText(canvas, "DISTRATOR", (box[0], max(25, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            draft = state["draft"]
            if draft:
                cv2.rectangle(canvas, draft[:2], draft[2:], (255, 255, 0), 2)
            if scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
            cv2.putText(canvas, f"{index + 1}/{len(pending)} | {row['sequence_id']} quadro {row['sequence_frame']} | {visibility}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(canvas, "Arraste caixa | B bola da jogada | D distrator | V visivel | P parcial | N nao visivel | U incerto", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 2)
            cv2.putText(canvas, "R limpa caixa atual | C limpa distratores | S salva | Q sair", (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(60)
            if key in {-1, 255}:
                continue
            char = chr(key & 0xFF).lower()
            if char in {"q", "\x1b"} or key == 27:
                save_csv(output, rows)
                cv2.destroyAllWindows()
                print(f"Progresso salvo: {output}")
                return 0
            if char in {"v", "p", "n", "u"}:
                visibility = {"v": "VISIBLE", "p": "PARTIAL", "n": "NOT_VISIBLE", "u": "UNCERTAIN"}[char]
                if char in {"n", "u"}:
                    target = None
                continue
            if char == "b" and state["draft"]:
                target = state["draft"]
                state["draft"] = None
                continue
            if char == "d" and state["draft"]:
                distractors.append(state["draft"])
                state["draft"] = None
                continue
            if char == "r":
                state["draft"] = None
                continue
            if char == "c":
                distractors = []
                continue
            if char == "s":
                if visibility in {"VISIBLE", "PARTIAL"} and target is None:
                    print("Marque a bola da jogada com uma caixa e pressione B antes de salvar.")
                    continue
                row["ball_visibility"] = visibility
                row["ball_of_play_box_json"] = json.dumps(target) if target else ""
                row["distractor_boxes_json"] = json.dumps(distractors)
                row["annotation_status"] = "completed"
                row["annotator"] = row.get("annotator") or "manual"
                save_csv(output, rows)
                print(f"Salvo: {row['sample_id']} ({index + 1}/{len(pending)})")
                break
    cv2.destroyAllWindows()
    print(f"Anotação concluída: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
