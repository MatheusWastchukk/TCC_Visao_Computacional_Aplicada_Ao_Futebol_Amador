"""Export all requested P05-P95 offsets per source and heldout TRAIN match."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent / "SOURCE-WINDOW-LOMO-EXP-0001")
    args = parser.parse_args()
    input_path = args.experiment_dir / "source_window_lomo_development_offsets_v1.csv"
    rows = read_csv(input_path)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["heldout_match"])].append(float(row["delta_goal_minus_candidate_seconds"]))
    if len(grouped) != 110:
        raise ValueError(f"Expected 5 sources x 22 heldout matches; got {len(grouped)} groups")
    quantiles = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    output = []
    for (source, heldout), values in sorted(grouped.items()):
        row: dict[str, object] = {"source": source, "heldout_match": heldout, "development_offset_count": len(values)}
        for q in quantiles:
            row[f"p{int(q*100):02d}_delta_seconds"] = f"{quantile(values, q):.3f}"
        row["p05_p95_window_seconds"] = f"{quantile(values, .95) - quantile(values, .05):.3f}"
        output.append(row)
    output_path = args.experiment_dir / "source_window_lomo_quantiles_v1.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    manifest_path = args.experiment_dir / "manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["all_reported_quantiles"] = list(quantiles)
    manifest["quantile_output"] = output_path.name
    manifest["methodological_limit"] = (
        "The supplied source scores are frozen OOF scores. For an outer heldout match, development-match selections may use scores from models trained on that heldout match. "
        "Therefore these are diagnostic sensitivity results, not leakage-free nested LOMO; regenerate source predictions within each outer fold before promotion."
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "offset_rows": len(rows), "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
