"""Resume limiares LOMO do Context Gate sem recalibrar ou acessar VALIDATION."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def point(rows: list[dict[str, str]], threshold: float) -> dict[str, object]:
    linked = [row for row in rows if row["goal_linked"] == "True"]
    negatives = [row for row in rows if row["context_decision"] == "REJECT_NO_ACTION"]
    keep = [row for row in rows if float(row["score_keep"]) >= threshold]
    return {
        "threshold": f"{threshold:.8f}", "kept_candidates": len(keep),
        "goal_linked_kept": sum(float(row["score_keep"]) >= threshold for row in linked),
        "goal_linked_total": len(linked),
        "negative_rejected": sum(float(row["score_keep"]) < threshold for row in negatives),
        "negative_total": len(negatives),
        "negative_rejection_rate": f"{sum(float(row['score_keep']) < threshold for row in negatives) / len(negatives):.6f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume os limiares out-of-fold do Context Gate em TRAIN.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    rows = read_csv(args.predictions)
    if not rows or any(row.get("goal_linked") not in {"True", "False"} for row in rows):
        raise ValueError("Predições incompletas.")
    linked_scores = [float(row["score_keep"]) for row in rows if row["goal_linked"] == "True"]
    if not linked_scores:
        raise ValueError("Não há candidatos ligados a gol no conjunto.")
    safety_threshold = min(linked_scores)
    thresholds = [0.0, safety_threshold, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    points = [point(rows, threshold) for threshold in sorted(set(thresholds))]
    args.output.mkdir(parents=True)
    fields = list(points[0])
    with (args.output / "threshold_points_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(points)
    safety = point(rows, safety_threshold)
    summary = {
        "scope": "somente TRAIN; limiares de predições leave-one-match-out já geradas",
        "safety_threshold": safety,
        "conclusion": "O menor limiar que preserva todos os candidatos ligados a gol não atinge rejeição suficiente para uso operacional.",
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
