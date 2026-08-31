"""Compara trackers temporais por partida e domínio de captura."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera comparação de trackers por partida.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for run in args.runs:
        with (run / "interval_tracking_results.csv").open(newline="", encoding="utf-8") as file:
            source = list(csv.DictReader(file))
        with (run / "frame_tracking.csv").open(newline="", encoding="utf-8") as file:
            frame_source = list(csv.DictReader(file))
        domains = {row["block_id"]: row["capture_domain"] for row in frame_source}
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in source:
            groups[(row["match_id"], domains.get(row["block_id"], ""))].append(row)
        for (match_id, domain), group in sorted(groups.items()):
            evaluable = [row for row in group if row["result"] != "inconclusive"]
            rows.append({
                "run": run.name, "match_id": match_id, "capture_domain": domain,
                "intervals": len(group), "evaluable": len(evaluable),
                "tp": sum(row["result"] == "true_positive" for row in evaluable),
                "tn": sum(row["result"] == "true_negative" for row in evaluable),
                "fp": sum(row["result"] == "false_positive" for row in evaluable),
                "fn": sum(row["result"] == "false_negative" for row in evaluable),
                "inconclusive": sum(row["result"] == "inconclusive" for row in group),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"grupos={len(rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
