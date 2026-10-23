"""Assemble the 25-match TRAIN net-motion matrix without replacing the 22-match source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_GROUPS = 5742
EXPECTED_MATCHES = 25
RECOVERED_MATCHES = {"match_02", "match_04", "match_06"}
NET_FIELDS = ("net_motion_inside_mean", "net_motion_ring_mean", "net_motion_ratio")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str], dict[str, str]]:
    result = {(row["match_id"], row["group_uid"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label} contains duplicate (match_id, group_uid) keys")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rich-dataset",
        type=Path,
        default=ROOT / "results/rich_temporal_dataset_train_fullpool_nested_v1/temporal_group_dataset_v1.csv",
    )
    parser.add_argument(
        "--historical-net-dataset",
        type=Path,
        default=ROOT / "results/net_motion_dataset_train_v1/temporal_group_dataset_net_motion_v1.csv",
    )
    parser.add_argument(
        "--historical-net-features",
        type=Path,
        default=ROOT / "results/net_motion_features_train_v1/net_motion_groups_v1.csv",
    )
    parser.add_argument(
        "--recovered-net-features",
        type=Path,
        default=ROOT / "results/net_motion_features_train_fullpool_recovered_v1/net_motion_groups_v1.csv",
    )
    parser.add_argument(
        "--goal-catalog",
        type=Path,
        default=ROOT / "results/local_trajectory_dataset_train_fullpool_v1/goal_timestamps_train_subset_v1.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/net_motion_dataset_train_fullpool_nested_v1",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    input_paths = {
        "rich_dataset": args.rich_dataset.resolve(),
        "historical_net_dataset": args.historical_net_dataset.resolve(),
        "historical_net_features": args.historical_net_features.resolve(),
        "recovered_net_features": args.recovered_net_features.resolve(),
        "goal_catalog": args.goal_catalog.resolve(),
    }
    missing_files = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(missing_files)

    rich_rows, rich_fields = read_csv(input_paths["rich_dataset"])
    historical_rows, historical_fields = read_csv(input_paths["historical_net_dataset"])
    historical_features, historical_feature_fields = read_csv(input_paths["historical_net_features"])
    recovered_features, recovered_feature_fields = read_csv(input_paths["recovered_net_features"])
    goals, goal_fields = read_csv(input_paths["goal_catalog"])

    expected_feature_fields = {"group_uid", "match_id", *NET_FIELDS}
    for label, fields in (
        ("historical features", historical_feature_fields),
        ("recovered features", recovered_feature_fields),
    ):
        if not expected_feature_fields.issubset(fields):
            raise ValueError(f"{label} is missing expected net-motion columns")
    if "group_uid" not in rich_fields or "match_id" not in rich_fields or "weak_label" not in rich_fields:
        raise ValueError("Rich dataset is missing its group key or weak label")

    rich = keyed(rich_rows, "rich dataset")
    historical = keyed(historical_rows, "historical net dataset")
    old_features = keyed(historical_features, "historical net features")
    recovered = keyed(recovered_features, "recovered net features")
    if set(historical) != set(old_features):
        raise ValueError("Historical net dataset and feature file have different group keys")
    if set(old_features) & set(recovered):
        raise ValueError("Recovered net features overlap the historical 22-match population")
    if {key[0] for key in recovered} != RECOVERED_MATCHES:
        raise ValueError(f"Recovered feature matches differ from {sorted(RECOVERED_MATCHES)}")
    if len(recovered) != 370:
        raise ValueError(f"Expected 370 recovered groups, found {len(recovered)}")
    if set(old_features) | set(recovered) != set(rich):
        raise ValueError("Historical plus recovered features do not exactly cover the rich dataset")
    match_counts = Counter(key[0] for key in rich)
    if len(rich) != EXPECTED_GROUPS or len(match_counts) != EXPECTED_MATCHES:
        raise ValueError(f"Expected {EXPECTED_GROUPS} groups/25 matches, found {len(rich)}/{len(match_counts)}")
    goal_matches = {row["match_id"] for row in goals if row.get("annotation_status") == "completed"}
    if (
        len(goals) != 152
        or sum(row.get("annotation_status") == "completed" for row in goals) != 152
        or goal_matches != set(match_counts)
    ):
        raise ValueError("Full-pool catalog must contain 152 completed timestamps across the 25 matrix matches")

    # Recomputed trajectory features must agree with the historical source for
    # every shared group before the old rows are retained verbatim.
    shared_fields = [field for field in rich_fields if field in historical_fields]
    for key, old_row in historical.items():
        new_row = rich[key]
        mismatched = [field for field in shared_fields if old_row.get(field, "") != new_row.get(field, "")]
        if mismatched:
            raise ValueError(f"Historical feature mismatch for {key}: {mismatched[:5]}")
        if old_row.get("weak_label") != new_row.get("weak_label"):
            raise ValueError(f"Historical weak label mismatch for {key}")

    rows: list[dict[str, str]] = []
    for source in rich_rows:
        key = (source["match_id"], source["group_uid"])
        if key in historical:
            row = dict(historical[key])
            for field in NET_FIELDS:
                if row.get(f"net_{field}", "") != old_features[key].get(field, ""):
                    raise ValueError(f"Historical net feature mismatch for {key}: {field}")
        else:
            feature = recovered[key]
            timestamp_delta = abs(
                float(feature["representative_timestamp_seconds"])
                - float(source["best_timestamp_seconds"])
            )
            if timestamp_delta > 1e-6:
                raise ValueError(f"Recovered representative timestamp mismatch for {key}")
            row = dict(source)
            for field in NET_FIELDS:
                row[f"net_{field}"] = feature[field]
        rows.append(row)

    output_fields = [*rich_fields, *(f"net_{field}" for field in NET_FIELDS)]
    if set(historical_fields) != set(output_fields):
        raise ValueError("Full-pool schema does not match the historical net dataset schema")
    output.mkdir(parents=True)
    dataset_path = output / "temporal_group_dataset_net_motion_v1.csv"
    with dataset_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "goal_timestamps_train_subset_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=goal_fields)
        writer.writeheader()
        writer.writerows(goals)

    summary = {
        "experiment_id": output.name,
        "scope": "TRAIN-only; full-pool matrix assembly; no model fitting and no V4 changes",
        "groups": len(rows),
        "matches": len(match_counts),
        "groups_by_match": dict(sorted(match_counts.items())),
        "historical_groups_retained_verbatim": len(historical),
        "recovered_groups": len(recovered),
        "recovered_matches": sorted(RECOVERED_MATCHES),
        "catalog_timestamps": len(goals),
        "completed_catalog_timestamps": sum(row.get("annotation_status") == "completed" for row in goals),
        "schema": output_fields,
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "output_sha256": sha256(dataset_path),
        "warning": "Dataset de preparação para Nested LOMO; não é avaliação nem alteração do pipeline operacional.",
    }
    (output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("experiment_id", "groups", "matches", "historical_groups_retained_verbatim", "recovered_groups", "catalog_timestamps", "output_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
