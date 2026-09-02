"""Revisa visualmente a evidência do tracker no instante decisivo de cada evento positivo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


FIELDS = ["event_id", "match_id", "sequence_id", "target_evidence", "tracker_evidence", "manual_review", "annotation_status"]


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
    cv2.putText(image, label, (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA)


def draw_polygon(image, points_json: str, color: tuple[int, int, int], label: str) -> None:
    if not points_json:
        return
    points = np.array(json.loads(points_json), dtype="int32")
    cv2.polylines(image, [points], True, color, 2)
    cv2.putText(image, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Revisa a evidência do tracker nos frames decisivos positivos.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--tracking-results", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--event-ids", nargs="+", default=None, help="Restringe a revisão aos eventos informados.")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    with args.sequences.open(newline="", encoding="utf-8") as file:
        sequences = {row["event_id"]: row for row in csv.DictReader(file) if row.get("is_event_frame") == "True"}
    with args.tracking_results.open(newline="", encoding="utf-8") as file:
        tracking = {(row["sequence_id"], row["sequence_frame"]): row for row in csv.DictReader(file)}
    with args.evidence.open(newline="", encoding="utf-8") as file:
        evidence = list(csv.DictReader(file))
    if args.event_ids:
        selected_ids = set(args.event_ids)
        evidence = [row for row in evidence if row["event_id"] in selected_ids]
        if not evidence:
            parser.error("Nenhuma evidência corresponde aos eventos informados.")
    existing: dict[str, dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["event_id"]: row for row in csv.DictReader(file)}
    rows = [existing.get(row["event_id"], {
        "event_id": row["event_id"], "match_id": row["match_id"], "sequence_id": row["sequence_id"],
        "target_evidence": row["target_evidence"], "tracker_evidence": row["tracker_evidence"],
        "manual_review": "", "annotation_status": "pending",
    }) for row in evidence]
    pending = [row for row in rows if not args.pending_only or row["annotation_status"] != "completed"]
    if not pending:
        print(f"Não há eventos pendentes: {args.output}")
        return 0
    window = "Revisão da evidência do evento"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    labels = {"s": "SUPORTA_EVENTO", "f": "FALHA_TRACKER", "u": "NAO_AVALIAVEL"}
    for index, review in enumerate(pending, start=1):
        sequence = sequences[review["event_id"]]
        tracker = tracking[(sequence["sequence_id"], sequence["sequence_frame"])]
        image = cv2.imread(sequence["image_path"])
        if image is None:
            raise FileNotFoundError(sequence["image_path"])
        draw_polygon(image, sequence.get("goal_region_points_json", ""), (0, 140, 255), "PLANO")
        draw_polygon(image, sequence.get("goal_volume_points_json", ""), (255, 160, 0), "VOLUME")
        if sequence["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            draw_box(image, sequence.get("ball_of_play_box_json", ""), (0, 220, 0), "ALVO")
        draw_box(image, tracker.get("selected_box_json", ""), (0, 220, 255), "TRACKER", 3)
        cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1200), 72), (0, 0, 0), -1)
        cv2.putText(image, f"{index}/{len(pending)} | {review['event_id']} | alvo={review['target_evidence']} | tracker={review['tracker_evidence']}", (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "S suporta evento | F falha do tracker | U nao avaliavel | Q sair", (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 120), 2, cv2.LINE_AA)
        scale = min(1.0, args.max_preview_width / image.shape[1])
        canvas = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
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
                review["manual_review"] = labels[char]
                review["annotation_status"] = "completed"
                save_csv(args.output, rows)
                print(f"Salvo: {review['event_id']} -> {labels[char]}")
                break
    cv2.destroyAllWindows()
    print(f"Revisão concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
