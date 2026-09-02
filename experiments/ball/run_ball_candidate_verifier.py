"""Treina um verificador visual para candidatas de bola propostas pelo detector."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


DEFAULT_LABELS = ["BOLA_JOGO", "BOLA_DISTRATORA", "PE_PERNA", "CABECA_TRONCO", "OUTRO"]


class CandidateDataset(Dataset):
    """Recorta a candidata com contexto visual, preservando a divisão por sequência."""

    def __init__(self, rows: list[dict[str, str]], labels: list[str], transform: v2.Compose) -> None:
        self.rows = rows
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (float(value) for value in json.loads(row["box_json"]))
        width, height = x2 - x1, y2 - y1
        padding = max(width, height) * 1.2
        left = max(0, round(x1 - padding))
        top = max(0, round(y1 - padding))
        right = min(image.shape[1], round(x2 + padding))
        bottom = min(image.shape[0], round(y2 + padding))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError(f"Crop inválido: {row['candidate_id']}")
        tensor = self.transform(torch.from_numpy(crop).permute(2, 0, 1))
        return tensor, self.labels.index(row["label"]), row["candidate_id"]


def macro_f1(targets: list[int], predictions: list[int], label_count: int) -> float:
    scores = []
    for label_index in range(label_count):
        tp = sum(target == label_index and prediction == label_index for target, prediction in zip(targets, predictions))
        fp = sum(target != label_index and prediction == label_index for target, prediction in zip(targets, predictions))
        fn = sum(target == label_index and prediction != label_index for target, prediction in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina o verificador visual de candidatas da bola.")
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--validation-sequences", nargs="+")
    split_group.add_argument("--validation-matches", nargs="+")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained-weights", type=Path, default=None, help="Checkpoint ResNet18 local a reutilizar.")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--binary-ball-likeness",
        action="store_true",
        help="Agrupa bola da jogada e bola distratora como BOLA; demais classes viram NAO_BOLA.",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = []
    for annotations_path in args.annotations:
        with annotations_path.open(newline="", encoding="utf-8") as file:
            rows.extend(row for row in csv.DictReader(file) if row["annotation_status"] == "completed" and row["label"] in DEFAULT_LABELS)
    labels = ["BOLA", "NAO_BOLA"] if args.binary_ball_likeness else DEFAULT_LABELS
    if args.binary_ball_likeness:
        rows = [{**row, "label": "BOLA" if row["label"] in {"BOLA_JOGO", "BOLA_DISTRATORA"} else "NAO_BOLA"} for row in rows]
    validation_ids = set(args.validation_sequences or [])
    validation_matches = set(args.validation_matches or [])
    if validation_matches:
        train_rows = [row for row in rows if row["match_id"] not in validation_matches]
        validation_rows = [row for row in rows if row["match_id"] in validation_matches]
    else:
        train_rows = [row for row in rows if row["sequence_id"] not in validation_ids]
        validation_rows = [row for row in rows if row["sequence_id"] in validation_ids]
    if not train_rows or not validation_rows:
        parser.error("A divisão por sequência produziu treino ou validação vazios.")
    train_labels, validation_labels = {row["label"] for row in train_rows}, {row["label"] for row in validation_rows}
    missing = set(labels) - train_labels
    if missing:
        parser.error(f"Classes ausentes no treino: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(args.output / "torch_hub"))
    train_transform = v2.Compose([
        v2.ToImage(), v2.Resize((128, 128), antialias=True), v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=ResNet18_Weights.DEFAULT.transforms().mean, std=ResNet18_Weights.DEFAULT.transforms().std),
    ])
    validation_transform = v2.Compose([
        v2.ToImage(), v2.Resize((128, 128), antialias=True), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=ResNet18_Weights.DEFAULT.transforms().mean, std=ResNet18_Weights.DEFAULT.transforms().std),
    ])
    train_loader = DataLoader(CandidateDataset(train_rows, labels, train_transform), batch_size=args.batch, shuffle=True, num_workers=0)
    validation_loader = DataLoader(CandidateDataset(validation_rows, labels, validation_transform), batch_size=args.batch, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    if args.pretrained_weights:
        if not args.pretrained_weights.exists():
            raise FileNotFoundError(args.pretrained_weights)
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(args.pretrained_weights, map_location="cpu", weights_only=True))
        pretrained_source = str(args.pretrained_weights)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        pretrained_source = ResNet18_Weights.DEFAULT.url
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.to(device)
    counts = Counter(row["label"] for row in train_rows)
    class_weights = torch.tensor([len(train_rows) / (len(labels) * counts[label]) for label in labels], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    best_f1, best_state, history = -1.0, None, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, targets, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        targets, predictions = [], []
        with torch.no_grad():
            for images, batch_targets, _ in validation_loader:
                batch_predictions = model(images.to(device)).argmax(dim=1).cpu().tolist()
                targets.extend(batch_targets.tolist())
                predictions.extend(batch_predictions)
        accuracy = sum(target == prediction for target, prediction in zip(targets, predictions)) / len(targets)
        f1 = macro_f1(targets, predictions, len(labels))
        history.append({"epoch": epoch, "validation_accuracy": round(accuracy, 6), "validation_macro_f1": round(f1, 6)})
        if f1 > best_f1:
            best_f1 = f1
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        print(f"época={epoch} acurácia_val={accuracy:.4f} f1_macro_val={f1:.4f}")
    torch.save({"state_dict": best_state, "labels": labels}, args.output / "best.pt")
    model.load_state_dict(best_state)
    model.eval()
    prediction_rows = []
    targets, predictions = [], []
    with torch.no_grad():
        for images, batch_targets, candidate_ids in validation_loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            batch_predictions = probabilities.argmax(dim=1).tolist()
            targets.extend(batch_targets.tolist())
            predictions.extend(batch_predictions)
            prediction_rows.extend({
                "candidate_id": candidate_id,
                "true_label": labels[target],
                "predicted_label": labels[prediction],
                "predicted_confidence": f"{float(probabilities[index, prediction]):.6f}",
                "ball_probability": f"{float(probabilities[index, labels.index('BOLA')]):.6f}" if "BOLA" in labels else "",
            } for index, (candidate_id, target, prediction) in enumerate(zip(candidate_ids, batch_targets.tolist(), batch_predictions)))
    class_metrics = {}
    for label_index, label in enumerate(labels):
        tp = sum(target == label_index and prediction == label_index for target, prediction in zip(targets, predictions))
        fp = sum(target != label_index and prediction == label_index for target, prediction in zip(targets, predictions))
        fn = sum(target == label_index and prediction != label_index for target, prediction in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_metrics[label] = {
            "support": sum(target == label_index for target in targets), "precision": round(precision, 6),
            "recall": round(recall, 6), "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
        }
    with (args.output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    threshold_rows = []
    if "BOLA" in labels:
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            decisions = [float(row["ball_probability"]) >= threshold for row in prediction_rows]
            true_ball = [row["true_label"] == "BOLA" for row in prediction_rows]
            ball_recall = sum(decision and truth for decision, truth in zip(decisions, true_ball)) / max(1, sum(true_ball))
            non_ball_rejection = sum((not decision) and (not truth) for decision, truth in zip(decisions, true_ball)) / max(1, sum(not truth for truth in true_ball))
            threshold_rows.append({"ball_threshold": f"{threshold:.2f}", "ball_recall": f"{ball_recall:.6f}", "non_ball_rejection": f"{non_ball_rejection:.6f}"})
        with (args.output / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(threshold_rows[0]))
            writer.writeheader()
            writer.writerows(threshold_rows)
    summary = {
        "experiment_id": args.output.name,
        "status": "completed", "base_model": "ResNet18 ImageNet", "pretrained_source": pretrained_source, "labels": labels,
        "task": "ball_likeness" if args.binary_ball_likeness else "five_class_candidate_type",
        "train_sequences": sorted({row["sequence_id"] for row in train_rows}),
        "validation_sequences": sorted(validation_ids), "validation_matches": sorted(validation_matches),
        "annotations": [str(path) for path in args.annotations],
        "train_count": len(train_rows), "validation_count": len(validation_rows),
        "train_labels": dict(Counter(row["label"] for row in train_rows)),
        "validation_labels": dict(Counter(row["label"] for row in validation_rows)),
        "class_metrics": class_metrics,
        "threshold_sweep": threshold_rows,
        "best_validation_macro_f1": round(best_f1, 6), "history": history,
        "execution_seconds": round(time.perf_counter() - started, 3),
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
