"""Prepara uma segunda amostra por partida para o verificador de candidatas."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def select_round_robin(rows: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    """Seleciona exemplos distribuídos pelas partidas disponíveis."""
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_match[row["match_id"]].append(row)
    for values in by_match.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        added = False
        for match_id in sorted(by_match):
            if by_match[match_id] and len(selected) < count:
                selected.append(by_match[match_id].pop())
                added = True
        if not added:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara candidatas V2 com cobertura por partida para o verificador.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--v1-sequences", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-contexts", type=int, default=60)
    parser.add_argument("--not-visible-contexts", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    with args.v1_sequences.open(newline="", encoding="utf-8") as file:
        v1_rows = list(csv.DictReader(file))
    used_ids = set()
    for row in v1_rows:
        match = re.search(r"origem=([^;]+)", row.get("notes", ""))
        if match:
            used_ids.add(match.group(1))
    with args.annotations.open(newline="", encoding="utf-8") as file:
        source_rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "train" and row["annotation_status"] == "completed" and row["sample_id"] not in used_ids
        ]
    visible_rows = [row for row in source_rows if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}]
    not_visible_rows = [row for row in source_rows if row["ball_visibility"] == "NOT_VISIBLE"]
    rng = random.Random(args.seed)
    selected = select_round_robin(visible_rows, args.visible_contexts, rng)
    selected.extend(select_round_robin(not_visible_rows, args.not_visible_contexts, rng))
    if len(selected) < args.visible_contexts + args.not_visible_contexts:
        raise ValueError("Não há frames suficientes após excluir os contextos da V1.")
    model = YOLO(str(args.model))
    output_rows: list[dict[str, str]] = []
    for context_index, row in enumerate(selected, start=1):
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        result = model.predict(source=image, conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        candidates = sorted(
            [(tuple(float(value) for value in box), float(confidence)) for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())],
            key=lambda item: item[1], reverse=True,
        )[: args.max_candidates]
        target = None
        if row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            target = tuple(float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"))
        sequence_id = f"verifier_v2_{context_index:03d}_{row['sample_id']}"
        for rank, (box, confidence) in enumerate(candidates, start=1):
            target_iou = iou(box, target) if target else 0.0
            automatic = target_iou >= 0.5
            output_rows.append({
                "candidate_id": f"{sequence_id}_candidate_{rank:02d}",
                "sequence_id": sequence_id, "sequence_frame": "1", "sample_id": row["sample_id"],
                "match_id": row["match_id"], "image_path": row["image_path"], "candidate_rank": str(rank),
                "confidence": f"{confidence:.6f}", "box_json": json.dumps(box), "target_iou": f"{target_iou:.6f}",
                "distractor_iou": "", "source_visibility": row["ball_visibility"],
                "label": "BOLA_JOGO" if automatic else "",
                "annotation_status": "completed" if automatic else "pending",
                "label_source": "caixa_humana_v2" if automatic else "revisao_manual",
                "notes": f"v2; contexto={context_index}; origem={row['sample_id']}",
            })
        print(f"[{context_index}/{len(selected)}] {row['sample_id']} candidatas={len(candidates)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    automatic = sum(row["annotation_status"] == "completed" for row in output_rows)
    print(f"contextos={len(selected)} candidatas={len(output_rows)} pre_rotuladas={automatic} pendentes={len(output_rows) - automatic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
