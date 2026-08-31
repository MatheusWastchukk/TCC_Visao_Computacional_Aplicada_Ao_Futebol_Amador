"""Lista intervalos cuja classificação mudou entre dois trackers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara mudanças entre trackers.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = {row["interval_id"]: row for row in read(args.base / "interval_tracking_results.csv")}
    candidate = {row["interval_id"]: row for row in read(args.candidate / "interval_tracking_results.csv")}
    rows: list[dict[str, str]] = []
    for interval_id, row in candidate.items():
        if base[interval_id]["result"] != row["result"]:
            rows.append({"interval_id": interval_id, "match_id": row["match_id"], "interval_label": row["interval_label"], "base_result": base[interval_id]["result"], "candidate_result": row["result"], "base_rate": base[interval_id]["appearance_rate"], "candidate_rate": row["appearance_rate"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"mudancas={len(rows)}")
    print(f"csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
