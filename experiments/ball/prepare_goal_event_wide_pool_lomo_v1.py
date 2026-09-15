"""Prepara o pool amplo temporal LOMO para avaliar um redutor visual em TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara top K temporal LOMO, somente TRAIN.")
    parser.add_argument("--rank-predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["train", "validation"], default="train")
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--min-separation-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    predictions = read_csv(args.rank_predictions)
    if "model" in predictions[0]:
        predictions = [row for row in predictions if row["model"] == "random_forest"]
    manifest = read_csv(args.split_manifest)
    videos = {row["match_id"]: row["video_filename"] for row in manifest if row["split"] == args.expected_split}
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        if row["match_id"] not in videos:
            raise AssertionError("As previsões devem conter somente o split esperado.")
        by_match[row["match_id"]].append(row)
    events: list[dict[str, object]] = []
    for match_id, rows in sorted(by_match.items()):
        selected: list[dict[str, str]] = []
        for row in sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True):
            center = float(row["best_timestamp_seconds"])
            if any(abs(center - float(other["best_timestamp_seconds"])) < args.min_separation_seconds for other in selected):
                continue
            selected.append(row)
            if len(selected) == args.top_k:
                break
        for rank, row in enumerate(selected, start=1):
            center = float(row["best_timestamp_seconds"])
            events.append({"event_id": f"{match_id}_wide_pool_{rank:03d}", "match_id": match_id, "video_filename": videos[match_id], "split": args.expected_split, "source_interval_id": row["group_uid"], "event_timestamp_seconds": f"{center:.6f}", "rank_score": row["rank_score"], "candidate_start_seconds": f"{max(0.0, center - 10.0):.6f}", "candidate_end_seconds": f"{center + 20.0:.6f}", "delivery_start_seconds": f"{max(0.0, center - 20.0):.6f}", "delivery_end_seconds": f"{center + 40.0:.6f}", "temporal_rank": rank})
    args.output.mkdir(parents=True)
    with (args.output / "wide_pool_candidates_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(events[0])); writer.writeheader(); writer.writerows(events)
    summary = {"experiment_id": args.output.name, "scope": f"somente {args.expected_split.upper()}", "top_k": args.top_k, "candidates": len(events), "matches": len(by_match), "analysis_window": "10+20", "delivery_window": "20+40"}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
