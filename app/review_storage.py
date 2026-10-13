"""Persistência retomável das decisões humanas de revisão."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MODES = {"alta_cobertura", "cobertura_reduzida", "triagem_rapida"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Lê um CSV UTF-8 com cabeçalho."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def safe_run_directory(results_root: Path, match_id: str, mode: str) -> Path:
    """Cria um destino determinístico, sempre dentro da raiz de resultados."""
    if mode not in MODES or not match_id.replace("_", "").isalnum():
        raise ValueError("Identificador de partida ou modo inválido.")
    root = results_root.resolve()
    destination = (root / f"{match_id}_{mode}").resolve()
    if root not in destination.parents:
        raise ValueError("A saída deve permanecer dentro da raiz de resultados.")
    return destination


def load_annotation_rows(manifest: Path, output: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Combina manifesto e decisões anteriores para permitir retomada."""
    source_rows = read_csv(manifest)
    if not source_rows:
        raise ValueError("Não há clipes para revisão.")
    existing = {row["segment_id"]: row for row in read_csv(output)} if output.is_file() else {}
    fields = list(source_rows[0])
    for field in ("human_label", "annotation_status", "notes"):
        if field not in fields:
            fields.append(field)
    rows = []
    for source in source_rows:
        previous = existing.get(source["segment_id"], {})
        row = {field: previous.get(field, source.get(field, "")) for field in fields}
        row["human_label"] = previous.get("human_label", "")
        row["annotation_status"] = previous.get("annotation_status", "pending")
        row["notes"] = previous.get("notes", "")
        rows.append(row)
    return rows, fields


def write_annotation_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Grava decisões humanas de forma atômica e separada da inferência."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Lê metadados locais de uma análise persistida."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("O arquivo de metadados da análise é inválido.")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Grava metadados em UTF-8 e substitui o arquivo de forma atômica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)
