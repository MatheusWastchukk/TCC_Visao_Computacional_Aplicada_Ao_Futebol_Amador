"""Seleciona limiar operacional em previsões LOMO, sem usar VALIDATION ou TEST."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Seleciona limiar do verificador de clipes com previsões TRAIN LOMO.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    with args.predictions.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("model", args.model) == args.model]
    if not rows:
        raise ValueError(f"Não há previsões para o modelo {args.model}.")
    labels = np.array([row["label"] == "GOAL" for row in rows])
    scores = np.array([float(row["score_goal"]) for row in rows])
    thresholds = sorted(set(np.concatenate(([0.0], scores, [1.0]))))
    sweep = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & labels)); fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels)); tn = int(np.sum(~predicted & ~labels))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        sweep.append({"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "recall": recall, "precision": precision, "false_positive_rate": fpr})
    eligible = [row for row in sweep if row["recall"] >= args.minimum_recall]
    if not eligible:
        raise ValueError("Nenhum limiar atende ao recall mínimo solicitado.")
    selected = max(eligible, key=lambda row: (row["precision"], -row["false_positive_rate"], row["threshold"]))
    args.output.mkdir(parents=True)
    with (args.output / "threshold_sweep_v1.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(sweep[0]); writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        writer.writerows([{**row, **{key: f"{value:.8f}" for key, value in row.items() if isinstance(value, float)}} for row in sweep])
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN", "model": args.model,
        "protocol": "limiar selecionado em previsões leave-one-match-out", "minimum_recall": args.minimum_recall,
        "selected": selected,
        "warning": "O limiar é decisão de desenvolvimento e precisa permanecer congelado antes de confirmação externa separada.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
