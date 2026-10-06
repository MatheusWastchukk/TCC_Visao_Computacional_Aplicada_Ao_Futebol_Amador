"""Freeze a match-held-out TRAIN fold and a single small-positive oversampling intervention."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "experiments/ball/ball_dataset_annotations_v2.csv"
ORACLE_MANIFEST = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001/oracle_pair_manifest_v1.csv"
DEFAULT_OUTPUT = ROOT / "experiments/ball/BALL-HARD-POSITIVE-EXP-0001"
HOLDOUT_SEED = 20261002
HOLDOUT_MATCH_COUNT = 5
SMALL_AREA_PX2 = 400.0
SMALL_POSITIVE_TOTAL_MULTIPLIER = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Sem linhas para gravar: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def area_px2(row: dict[str, str]) -> float:
    if row.get("ball_visibility") not in {"VISIBLE", "PARTIAL"}:
        return 0.0
    return (float(row["ball_x2"]) - float(row["ball_x1"])) * (float(row["ball_y2"]) - float(row["ball_y1"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--oracle-manifest", type=Path, default=ORACLE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Saída já contém arquivos: {args.output}")

    rows = read_csv(args.annotations)
    if not rows or not {"sample_id", "match_id", "split", "ball_visibility", "image_path", "ball_x1", "ball_y1", "ball_x2", "ball_y2"} <= set(rows[0]):
        raise ValueError("Schema do dataset BALL v2 mudou; confira antes de congelar o fold.")
    train_rows = [r for r in rows if r["split"].lower() == "train"]
    if len(train_rows) != 750 or len({r["match_id"] for r in train_rows}) != 25:
        raise ValueError(f"Escopo esperado 750 amostras TRAIN/25 partidas; encontrado {len(train_rows)}/{len({r['match_id'] for r in train_rows})}")
    oracle_matches = {r["match_id"] for r in read_csv(args.oracle_manifest)}
    all_matches = sorted({r["match_id"] for r in train_rows})
    eligible = sorted(set(all_matches) - oracle_matches)
    if len(eligible) < HOLDOUT_MATCH_COUNT:
        raise ValueError("Não há partidas TRAIN independentes do piloto oracle em número suficiente.")
    rng = random.Random(HOLDOUT_SEED)
    holdout_matches = sorted(rng.sample(eligible, HOLDOUT_MATCH_COUNT))
    train_matches = sorted(set(all_matches) - set(holdout_matches))
    train_fold = [r for r in train_rows if r["match_id"] in train_matches]
    holdout = [r for r in train_rows if r["match_id"] in holdout_matches]

    baseline_rows: list[dict[str, object]] = [{**r, "fold_role": "train", "augmentation_copy": "original"} for r in train_fold]
    variant_rows: list[dict[str, object]] = []
    small_count = 0
    for row in train_fold:
        variant_rows.append({**row, "fold_role": "train", "augmentation_copy": "original"})
        if area_px2(row) < SMALL_AREA_PX2 and row["ball_visibility"] in {"VISIBLE", "PARTIAL"}:
            small_count += 1
            for copy_index in range(1, SMALL_POSITIVE_TOTAL_MULTIPLIER):
                duplicate = dict(row)
                duplicate["sample_id"] = f"{row['sample_id']}__smallpos_x{copy_index + 1}"
                duplicate["notes"] = (row.get("notes", "") + f" | fold_oversample=small_positive_lt_{int(SMALL_AREA_PX2)}px2; copy={copy_index + 1}").strip(" |")
                duplicate["fold_role"] = "train"
                duplicate["augmentation_copy"] = f"small_positive_duplicate_{copy_index}"
                variant_rows.append(duplicate)

    holdout_rows: list[dict[str, object]] = [{**r, "split": "holdout_train_match", "fold_role": "evaluation", "augmentation_copy": "original"} for r in holdout]
    args.output.mkdir(parents=True)
    write_csv(args.output / "train_baseline_annotations_v1.csv", baseline_rows)
    write_csv(args.output / "train_small_positive_oversampled_annotations_v1.csv", variant_rows)
    write_csv(args.output / "holdout_train_annotations_v1.csv", holdout_rows)
    write_csv(args.output / "holdout_matches_v1.csv", [{"match_id": match, "selection_seed": HOLDOUT_SEED, "selection_basis": "random sample from TRAIN matches absent from oracle_pair_manifest_v1.csv"} for match in holdout_matches])

    def counts(data: list[dict[str, object]]) -> dict[str, object]:
        rows_by_vis = Counter(str(r["ball_visibility"]) for r in data)
        rows_by_match = Counter(str(r["match_id"]) for r in data)
        positives = [r for r in data if str(r["ball_visibility"]) in {"VISIBLE", "PARTIAL"}]
        small = sum(area_px2({k: str(v) for k, v in r.items()}) < SMALL_AREA_PX2 for r in positives)
        return {"samples": len(data), "matches": len(rows_by_match), "visibility": dict(sorted(rows_by_vis.items())),
                "positive_visible_partial": len(positives), "positive_box_area_lt_400px2": small,
                "per_match_samples": dict(sorted(rows_by_match.items()))}

    summary = {
        "experiment_id": args.output.name,
        "source_scope": "Existing TRAIN annotations only; original TRAIN split has 25 matches; no VALIDATION or TEST rows included.",
        "inputs": {"annotations": str(args.annotations.relative_to(ROOT)), "annotations_sha256": file_sha256(args.annotations),
                   "oracle_manifest": str(args.oracle_manifest.relative_to(ROOT)), "oracle_manifest_sha256": file_sha256(args.oracle_manifest)},
        "split": {"seed": HOLDOUT_SEED, "holdout_matches": holdout_matches, "train_matches": train_matches,
                  "oracle_matches_excluded_from_holdout_eligibility": sorted(oracle_matches),
                  "holdout_is_disjoint_from_oracle_matches": not (set(holdout_matches) & oracle_matches)},
        "intervention": {"only_change": "Oversample existing VISIBLE/PARTIAL ball boxes with area <400 px2 to 3 total copies.",
                         "area_threshold_px2": SMALL_AREA_PX2, "original_small_positive_count_in_train_fold": small_count,
                         "extra_duplicate_rows": small_count * (SMALL_POSITIVE_TOTAL_MULTIPLIER - 1),
                         "total_multiplier": SMALL_POSITIVE_TOTAL_MULTIPLIER,
                         "all_other_labels_and_model_settings": "unchanged"},
        "baseline_train": counts(baseline_rows),
        "oversampled_train": counts(variant_rows),
        "holdout": counts(holdout_rows),
        "comparison_protocol": {"init_weights": "yolo11n.pt for both", "epochs": 30, "imgsz": 960,
                                 "batch": 8, "seed": 0, "device": "CPU .venv; Python 3.11.9; PyTorch 2.13.0+cpu; Ultralytics 8.4.102",
                                 "validation_during_training": False, "checkpoint_choice": "last.pt at fixed epoch; no holdout metric used to select weights",
                                 "candidate_recall": "any top-10 prediction at confidence 0.01 with IoU >=0.50 on VISIBLE/PARTIAL holdout frames",
                                 "also_report": ["recall by match and visibility", "proposals per frame", "cost", "per-match variation"],
                                 "tracking_reopen_gate": "retest the same Viterbi only if oversampled holdout positive candidate recall is >=0.50 and exceeds the baseline on the same 5 matches; otherwise NO-GO tracker"},
        "warning": "The five evaluation matches are a pre-fixed holdout from TRAIN, not VALIDATION or TEST and not an estimate of external generalization. This holdout supports one detector engineering comparison only.",
    }
    (args.output / "protocol_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = (
        "# BALL Hard Positive Fold EXP-0001\n\n"
        "This freezes a TRAIN-only match split for one detector intervention. The five holdout matches were sampled with seed `20261002` from TRAIN matches absent from the 20-pair oracle manifest. `VALIDATION` and `TEST` annotations are excluded.\n\n"
        f"**Holdout matches:** {', '.join(holdout_matches)}  \n"
        f"**Training matches:** {len(train_matches)}  \n"
        f"**Baseline train:** {len(baseline_rows)} annotated rows  \n"
        f"**Oversampled train:** {len(variant_rows)} rows; {small_count} small positives get two extra copies  \n"
        f"**Evaluation:** {len(holdout_rows)} rows / {len(holdout_matches)} untouched TRAIN matches\n\n"
        "The only intervention is 3x sampling for existing VISIBLE/PARTIAL boxes with area below 400 px². Both models will start from `yolo11n.pt`, use the same fixed training configuration and fixed epoch count, and select `last.pt`; no holdout metric will select weights. This is a development comparison, not a validation or generalization claim.\n\n"
        "The split remains frozen even if the small-positive distribution is unbalanced. See `protocol_v1.json` and the three annotation CSVs for counts and exact sample membership.\n\n"
        "Pre-registered tracker gate: retest the same Viterbi only if oversampled holdout positive candidate recall is at least 50% and exceeds baseline on the same five matches. Otherwise the tracker stays NO-GO; this engineering gate is not a universal scientific threshold.\n"
    )
    (args.output / "README.md").write_text(md, encoding="utf-8")
    print(json.dumps({"experiment_id": args.output.name, "holdout_matches": holdout_matches,
                      "train_matches": len(train_matches), "baseline_rows": len(baseline_rows),
                      "oversampled_rows": len(variant_rows), "small_positive_count": small_count,
                      "holdout_rows": len(holdout_rows), "overlap_with_oracle": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
