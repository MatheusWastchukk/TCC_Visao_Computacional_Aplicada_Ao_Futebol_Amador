"""Compare frozen V4 modes against Match 43 catalog timestamps; no fitting/tuning."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.events.operational_modes_v4 import analyze_match
EXP = ROOT / "experiments/ball/BALL-HIGH-RES-CONFIRMATION-EXP-0001-MATCH43"
VARIANTS = {
    "global_960": EXP / "pipeline_global960_v1",
    "global_1536": EXP / "pipeline_global1536_v1",
    "hybrid_960_roi_1536": EXP / "pipeline_hybrid960_roi1536_v1",
}
GOALS = {
    "match43_goal_1": (3422.0, "00:57:02"),
    "match43_goal_2": (3546.0, "00:59:06"),
    "match43_goal_3": (3841.0, "01:04:01"),
    "match43_goal_4": (3930.0, "01:05:30"),
}
MODES = ("alta_cobertura", "cobertura_reduzida", "triagem_rapida")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def first_cover(rows, start_key, end_key, timestamp):
    for index, row in enumerate(rows, 1):
        if float(row[start_key]) <= timestamp <= float(row[end_key]):
            return index, row
    return None, None


def main():
    summary_rows = []
    coverage_rows = []
    detailed = {}
    for variant, base in VARIANTS.items():
        wide = read_csv(base / "wide_pool_scored.csv")
        mining = json.loads((base / "stages/mining/metrics.json").read_text(encoding="utf-8"))
        by_match = mining["by_match"]["match_43"]
        detailed[variant] = {}
        for mode in MODES:
            result = analyze_match(wide, mode)
            selected, segments = result["selected_candidates"], result["segments"]
            assert isinstance(selected, list) and isinstance(segments, list)
            goal_details = []
            for goal_id, (timestamp, hhmmss) in GOALS.items():
                pool_rank, pool_row = first_cover(wide, "delivery_start_seconds", "delivery_end_seconds", timestamp)
                selected_rank, selected_row = first_cover(selected, "delivery_start_seconds", "delivery_end_seconds", timestamp)
                segment_rank, segment_row = first_cover(segments, "start_seconds", "end_seconds", timestamp)
                goal_details.append({
                    "goal_id": goal_id, "timestamp_seconds": timestamp, "timestamp_hhmmss": hhmmss,
                    "candidate_pool_covered": pool_row is not None, "candidate_pool_rank": pool_rank,
                    "candidate_pool_event_id": pool_row.get("event_id") if pool_row else "",
                    "selected_covered": selected_row is not None, "selected_rank": selected_rank,
                    "selected_event_id": selected_row.get("event_id") if selected_row else "",
                    "segment_covered": segment_row is not None, "segment_rank": segment_rank,
                    "segment_id": segment_row.get("segment_id") if segment_row else "",
                })
                coverage_rows.append({"variant": variant, "mode": mode, **goal_details[-1]})
            row = {
                "variant": variant, "mode": mode,
                "raw_hits": by_match["hits"], "temporal_groups": by_match["groups"],
                "selected_groups_max60": by_match["selected"], "wide_pool_size": len(wide),
                "candidate_pool_goal_recall": sum(g["candidate_pool_covered"] for g in goal_details),
                "goals_total": len(GOALS), "selected_candidate_count": len(selected),
                "selected_goal_recall": sum(g["selected_covered"] for g in goal_details),
                "segment_count": len(segments),
                "segment_goal_recall": sum(g["segment_covered"] for g in goal_details),
                "unique_review_seconds": round(float(result["summary"]["unique_review_seconds"]), 3),
                "goal_details": goal_details,
            }
            summary_rows.append({k: v for k, v in row.items() if k != "goal_details"})
            detailed[variant][mode] = row

            if mode == "cobertura_reduzida":
                saved = json.loads((base / "results.json").read_text(encoding="utf-8"))
                if result["summary"] != saved["summary"]:
                    raise AssertionError(f"Offline V4 recomputation differs from saved run: {variant}")

    OUT = EXP / "pipeline_impact_summary_v1"
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("mode_comparison_v1.csv", summary_rows), ("goal_coverage_v1.csv", coverage_rows)):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "status": "completed",
        "scope": "Match 43 only; four catalogued timestamps; candidate generation/ranking runs are frozen; V4 modes recomputed offline without fitting or tuning",
        "goal_timestamps": {k: {"seconds": v[0], "hhmmss": v[1]} for k, v in GOALS.items()},
        "modes": detailed,
        "limitations": [
            "One external match only; no estimate across matches.",
            "Catalog timestamps are the reference; this does not certify an official goal decision.",
            "Candidate pool recall means a candidate delivery window covers the timestamp.",
            "TEST was not accessed.",
        ],
    }
    (OUT / "summary_v1.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
