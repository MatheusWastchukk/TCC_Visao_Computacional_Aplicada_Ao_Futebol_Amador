"""Resolve a auditoria de negativos temporais e materializa o manifesto V2 auditado."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve auditoria humana de negativos do dataset V2.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-no-goal-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.all_no_goal_confirmed:
        raise ValueError("Use --all-no-goal-confirmed somente após confirmação humana explícita.")
    if args.output.exists():
        raise FileExistsError(f"A saída já existe: {args.output}")
    dataset, fields = read_csv(args.dataset)
    audit, audit_fields = read_csv(args.audit)
    required = {"event_id", "expected_label", "annotation_status"}
    if missing := required - set(audit_fields):
        raise ValueError(f"Auditoria sem colunas obrigatórias: {sorted(missing)}")
    if any(row.get("split") != "train" for row in dataset + audit):
        raise AssertionError("A resolução aceita somente TRAIN; VALIDATION e TEST permanecem fechados.")
    if any(row["annotation_status"] != "completed" for row in audit):
        raise ValueError("A auditoria possui itens não concluídos.")
    if any(row["expected_label"] != "NO_GOAL" for row in audit):
        raise ValueError("Esta resolução foi projetada somente para a fila de negativos temporais.")
    resolved_fields = list(audit_fields)
    if "ground_truth" not in resolved_fields:
        resolved_fields.append("ground_truth")
    if "resolution_note" not in resolved_fields:
        resolved_fields.append("resolution_note")
    resolved = []
    for row in audit:
        current = {field: row.get(field, "") for field in resolved_fields}
        current["ground_truth"] = "NO_GOAL"
        current["resolution_note"] = "Rótulo reconstruído após confirmação humana explícita de que os 19 itens eram NO_GOAL; o anotador anterior não persistiu ground_truth."
        resolved.append(current)
    audited_ids = {row["event_id"] for row in resolved}
    audited_dataset = []
    for row in dataset:
        current = dict(row)
        if current["clip_id"] in audited_ids:
            current["review_status"] = "human_confirmed_reconstructed"
            current["notes"] += " Auditoria humana: NO_GOAL; rótulo reconstruído após falha de persistência documentada."
        audited_dataset.append(current)
    args.output.mkdir(parents=True)
    write_csv(args.output / "negative_audit_queue_v1_resolved.csv", resolved, resolved_fields)
    write_csv(args.output / "goal_clip_verifier_dataset_v2_audited.csv", audited_dataset, fields)
    summary = {
        "experiment_id": args.output.name, "scope": "somente TRAIN", "dataset_clips": len(audited_dataset),
        "labels": dict(Counter(row["label"] for row in audited_dataset)), "audit_items": len(resolved),
        "audit_labels": dict(Counter(row["ground_truth"] for row in resolved)),
        "resolution_method": "confirmação humana explícita após falha de persistência do anotador",
        "remaining_limitation": "A auditoria é amostral: 19 dos 78 negativos temporais seguros foram revisados."
    }
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
