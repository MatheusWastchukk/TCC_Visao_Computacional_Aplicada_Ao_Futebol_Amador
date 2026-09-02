"""Executa um piloto de múltiplas hipóteses em sequências anotadas de treino."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def transition_penalty(previous: tuple[float, float, float, float], current: tuple[float, float, float, float], diagonal: float) -> float:
    """Penaliza saltos e mudanças de escala sem impedir trajetórias rápidas."""
    previous_center, current_center = center(previous), center(current)
    distance = math.hypot(previous_center[0] - current_center[0], previous_center[1] - current_center[1]) / diagonal
    previous_area = max(1.0, (previous[2] - previous[0]) * (previous[3] - previous[1]))
    current_area = max(1.0, (current[2] - current[0]) * (current[3] - current[1]))
    size_change = abs(math.log(current_area / previous_area))
    return 4.0 * distance + 0.25 * size_change


def ball_probability(image: np.ndarray, box: tuple[float, float, float, float], model, transform, device) -> float:
    """Calcula P(BOLA) para uma candidata usando o mesmo crop do verificador."""
    x1, y1, x2, y2 = box
    padding = max(x2 - x1, y2 - y1) * 1.2
    left, top = max(0, round(x1 - padding)), max(0, round(y1 - padding))
    right, bottom = min(image.shape[1], round(x2 + padding)), min(image.shape[0], round(y2 + padding))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    return float(probabilities[0].item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia múltiplas hipóteses de tracking da bola em sequências de treino.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-matches", nargs="+", default=None, help="Restringe a avaliação às partidas informadas.")
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verifier-checkpoint", type=Path, default=None, help="Checkpoint binário para adicionar P(BOLA) ao score.")
    parser.add_argument("--verifier-weight", type=float, default=0.0, help="Peso de P(BOLA) no score; 0 preserva o baseline.")
    args = parser.parse_args()

    with args.sequences.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if args.include_matches:
        matches = set(args.include_matches)
        rows = [row for row in rows if row["match_id"] in matches]
        if not rows:
            parser.error("Nenhuma sequência corresponde às partidas informadas.")
    rows_by_sequence: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_sequence.setdefault(row["sequence_id"], []).append(row)
    model = YOLO(str(args.model))
    verifier_model = None
    verifier_transform = None
    verifier_device = None
    if args.verifier_checkpoint:
        if args.verifier_weight <= 0:
            parser.error("--verifier-weight deve ser maior que zero quando houver checkpoint.")
        checkpoint = torch.load(args.verifier_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("labels") != ["BOLA", "NAO_BOLA"]:
            parser.error("O checkpoint deve ser um verificador binário com classes BOLA e NAO_BOLA.")
        verifier_model = resnet18(weights=None)
        verifier_model.fc = torch.nn.Linear(verifier_model.fc.in_features, 2)
        verifier_model.load_state_dict(checkpoint["state_dict"])
        verifier_device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
        verifier_model.to(verifier_device).eval()
        verifier_transform = v2.Compose([
            v2.ToImage(), v2.Resize((128, 128), antialias=True), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=ResNet18_Weights.DEFAULT.transforms().mean, std=ResNet18_Weights.DEFAULT.transforms().std),
        ])
    args.output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, str]] = []
    trajectory_rows: list[dict[str, str]] = []

    for sequence_id, sequence_rows in rows_by_sequence.items():
        sequence_rows.sort(key=lambda row: int(row["sequence_frame"]))
        candidates_by_frame: list[list[tuple[tuple[float, float, float, float], float, float]]] = []
        image_diagonal = 1.0
        for row in sequence_rows:
            image = cv2.imread(row["image_path"])
            if image is None:
                raise FileNotFoundError(row["image_path"])
            image_diagonal = math.hypot(image.shape[0], image.shape[1])
            result = model.predict(source=image, conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            candidates = sorted([
                (tuple(float(value) for value in box), float(confidence))
                for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
            ], key=lambda item: item[1], reverse=True)[: args.max_candidates]
            candidates_by_frame.append([
                (box, confidence, ball_probability(image, box, verifier_model, verifier_transform, verifier_device) if verifier_model else 0.0)
                for box, confidence in candidates
            ])

        scores: list[list[float]] = []
        backpointers: list[list[int]] = []
        for frame_index, candidates in enumerate(candidates_by_frame):
            if not candidates:
                scores.append([])
                backpointers.append([])
                continue
            if frame_index == 0 or not scores[-1]:
                scores.append([confidence + args.verifier_weight * probability for _, confidence, probability in candidates])
                backpointers.append([-1] * len(candidates))
                continue
            current_scores: list[float] = []
            current_backpointers: list[int] = []
            for candidate_box, confidence, probability in candidates:
                options = [
                    previous_score - transition_penalty(candidates_by_frame[frame_index - 1][previous_index][0], candidate_box, image_diagonal)
                    for previous_index, previous_score in enumerate(scores[-1])
                ]
                best_previous = max(range(len(options)), key=options.__getitem__)
                current_scores.append(confidence + args.verifier_weight * probability + options[best_previous])
                current_backpointers.append(best_previous)
            scores.append(current_scores)
            backpointers.append(current_backpointers)

        available_frames = [index for index, values in enumerate(scores) if values]
        selected: dict[int, int] = {}
        if available_frames:
            final_frame = available_frames[-1]
            candidate_index = max(range(len(scores[final_frame])), key=scores[final_frame].__getitem__)
            for frame_index in range(final_frame, -1, -1):
                if candidate_index < 0 or not candidates_by_frame[frame_index]:
                    break
                selected[frame_index] = candidate_index
                candidate_index = backpointers[frame_index][candidate_index]

        correct_frames = wrong_switches = candidate_present = target_frames = 0
        for frame_index, row in enumerate(sequence_rows):
            target_box = (
                tuple(json.loads(row["ball_of_play_box_json"]))
                if row.get("ball_visibility", "VISIBLE") in {"VISIBLE", "PARTIAL"} and row["ball_of_play_box_json"]
                else None
            )
            distractors = [tuple(box) for box in json.loads(row["distractor_boxes_json"] or "[]")]
            candidates = candidates_by_frame[frame_index]
            target_rank = next((index + 1 for index, (box, _, _) in enumerate(candidates) if target_box and iou(box, target_box) >= 0.5), 0)
            selected_index = selected.get(frame_index, -1)
            selected_box = candidates[selected_index][0] if selected_index >= 0 else None
            selected_confidence = candidates[selected_index][1] if selected_index >= 0 else 0.0
            selected_ball_probability = candidates[selected_index][2] if selected_index >= 0 else 0.0
            target_iou = iou(selected_box, target_box) if selected_box and target_box else 0.0
            distractor_iou = max((iou(selected_box, distractor) for distractor in distractors), default=0.0) if selected_box else 0.0
            if target_box:
                target_frames += 1
                candidate_present += bool(target_rank)
                correct_frames += target_iou >= 0.5
                wrong_switches += distractor_iou >= 0.5
            frame_rows.append({
                "sequence_id": sequence_id,
                "sequence_frame": row["sequence_frame"],
                "sample_id": row["sample_id"],
                "candidate_count": str(len(candidates)),
                "target_annotated": str(target_box is not None),
                "target_candidate_rank": str(target_rank),
                "selected_candidate_rank": str(selected_index + 1 if selected_index >= 0 else 0),
                "selected_confidence": f"{selected_confidence:.6f}",
                "selected_ball_probability": f"{selected_ball_probability:.6f}",
                "selected_box_json": json.dumps(selected_box) if selected_box else "",
                "selected_target_iou": f"{target_iou:.6f}",
                "selected_distractor_iou": f"{distractor_iou:.6f}",
                "identity_correct": str(target_iou >= 0.5),
                "wrong_ball_switch": str(distractor_iou >= 0.5),
            })
        trajectory_rows.append({
            "sequence_id": sequence_id,
            "target_frames": str(target_frames),
            "target_candidate_recall": f"{candidate_present / target_frames:.6f}" if target_frames else "",
            "identity_accuracy": f"{correct_frames / target_frames:.6f}" if target_frames else "",
            "wrong_ball_switch_rate": f"{wrong_switches / target_frames:.6f}" if target_frames else "",
        })
        print(f"Processada: {sequence_id}")

    for filename, output_rows in (("frame_identity_results.csv", frame_rows), ("trajectory_identity_summary.csv", trajectory_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    target_rows = [row for row in frame_rows if row["target_annotated"] == "True"]
    metrics = {
        "experiment_id": args.output.name,
        "sequences": len(trajectory_rows),
        "target_frames": len(target_rows),
        "candidate_recall": sum(row["target_candidate_rank"] != "0" for row in target_rows) / len(target_rows),
        "identity_accuracy": sum(row["identity_correct"] == "True" for row in target_rows) / len(target_rows),
        "wrong_ball_switch_rate": sum(row["wrong_ball_switch"] == "True" for row in target_rows) / len(target_rows),
        "confidence": args.confidence,
        "max_candidates": args.max_candidates,
        "verifier_checkpoint": str(args.verifier_checkpoint) if args.verifier_checkpoint else None,
        "verifier_weight": args.verifier_weight,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
