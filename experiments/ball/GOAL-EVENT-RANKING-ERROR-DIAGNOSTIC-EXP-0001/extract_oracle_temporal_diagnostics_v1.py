"""Explore fixed temporal diagnostics from the reviewed oracle annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


EXP = Path(__file__).resolve().parent
DEFAULT_INPUT = EXP / "oracle_frame_annotations_v1_temporal_geometry.csv"
DEFAULT_FRAMES_OUT = EXP / "oracle_temporal_diagnostic_frames_v1.csv"
DEFAULT_SEQ_OUT = EXP / "oracle_temporal_diagnostic_sequences_v1.csv"
DEFAULT_PAIRS_OUT = EXP / "oracle_temporal_diagnostic_pairs_v1.csv"
DEFAULT_SUMMARY = EXP / "oracle_temporal_diagnostics_summary_v1.md"
WINDOW = set(range(16, 27))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No output rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box_center(raw: str) -> tuple[float, float] | None:
    if not raw:
        return None
    x1, y1, x2, y2 = map(float, json.loads(raw))
    return (x1 + x2) / 2, (y1 + y2) / 2


def mean_side(points: list[list[float]]) -> float:
    return sum(math.dist(points[i], points[(i + 1) % 4]) for i in range(4)) / 4


def inside(point: tuple[float, float], points: list[list[float]]) -> bool:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def longest_run(indices: list[int]) -> int:
    best = current = 0
    previous: int | None = None
    for index in indices:
        current = current + 1 if previous is not None and index == previous + 1 else 1
        best = max(best, current)
        previous = index
    return best


def pair_result(pos: object, neg: object, higher_is_better: bool) -> str:
    if pos == "" or neg == "":
        return "NOT_EVALUABLE"
    p, n = float(pos), float(neg)
    if abs(p - n) <= 1e-9:
        return "TIE"
    positive_wins = p > n if higher_is_better else p < n
    return "POSITIVE_WIN" if positive_wins else "NEGATIVE_WIN"


def extract(input_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, Counter[str]]]:
    all_rows = load_csv(input_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        if int(row["sequence_frame"]) in WINDOW:
            grouped.setdefault(row["sequence_id"], []).append(row)

    frame_out: list[dict[str, object]] = []
    seq_out: list[dict[str, object]] = []
    values: dict[str, dict[str, object]] = {}
    for sid, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["sequence_frame"]))
        posneg = sid.rsplit("_", 1)[1]
        observations: dict[int, dict[str, object]] = {}
        for row in rows:
            idx = int(row["sequence_frame"])
            loc = box_center(row.get("ball_of_play_box_json", ""))
            aperture_ok = row["ball_visibility"] == "VISIBLE" and loc is not None and row["effective_goal_region_status"] == "VALID" and row["goal_region_geometry_inside_frame"] == "True"
            volume_ok = row["ball_visibility"] == "VISIBLE" and loc is not None and row["effective_goal_volume_status"] == "VALID" and row["goal_volume_geometry_inside_frame"] == "True"
            dist_center: float | None = None
            in_aperture: bool | None = None
            in_volume: bool | None = None
            if aperture_ok:
                aperture = json.loads(row["effective_goal_region_points_json"])
                center = (sum(point[0] for point in aperture) / 4, sum(point[1] for point in aperture) / 4)
                side = mean_side(aperture)
                if side <= 0 or not math.isfinite(side):
                    raise ValueError(f"Degenerate aperture {sid} frame {idx}")
                dist_center = math.dist(loc, center) / side
                in_aperture = inside(loc, aperture)
            if volume_ok:
                volume = json.loads(row["effective_goal_volume_points_json"])
                in_volume = inside(loc, volume)
            observations[idx] = {
                "row": row, "loc": loc, "aperture_ok": aperture_ok, "volume_ok": volume_ok,
                "distance": dist_center, "inside_aperture": in_aperture, "inside_volume": in_volume,
            }
            frame_out.append({
                "pair_id": sid.rsplit("_", 1)[0], "sequence_id": sid, "class": posneg,
                "sequence_frame": idx, "ball_visibility": row["ball_visibility"],
                "aperture_eligible": aperture_ok, "normalized_distance_to_aperture_center": "" if dist_center is None else round(dist_center, 8),
                "center_inside_aperture": "" if in_aperture is None else in_aperture,
                "volume_eligible": volume_ok, "center_inside_volume": "" if in_volume is None else in_volume,
            })

        aperture_indices = [i for i, obs in observations.items() if obs["aperture_ok"]]
        volume_indices = [i for i, obs in observations.items() if obs["volume_ok"]]
        changes: list[float] = []
        enter_count = exit_count = 0
        for idx in range(16, 26):
            first, second = observations[idx], observations[idx + 1]
            if first["aperture_ok"] and second["aperture_ok"]:
                changes.append(float(second["distance"]) - float(first["distance"]))
                if first["inside_aperture"] is False and second["inside_aperture"] is True:
                    enter_count += 1
                elif first["inside_aperture"] is True and second["inside_aperture"] is False:
                    exit_count += 1

        inside_aperture_indices = [i for i in aperture_indices if observations[i]["inside_aperture"]]
        inside_volume_indices = [i for i in volume_indices if observations[i]["inside_volume"]]
        mean_change = sum(changes) / len(changes) if changes else None
        approach_fraction = sum(value < -1e-9 for value in changes) / len(changes) if changes else None
        sequence_values: dict[str, object] = {
            "pair_id": sid.rsplit("_", 1)[0], "sequence_id": sid, "class": posneg,
            "aperture_eligible_frames": len(aperture_indices), "consecutive_aperture_pairs": len(changes),
            "mean_distance_change_per_0_2s_step": "" if mean_change is None else round(mean_change, 8),
            "fraction_steps_approaching_center": "" if approach_fraction is None else round(approach_fraction, 6),
            "aperture_entries_2d": enter_count, "aperture_exits_2d": exit_count,
            "max_consecutive_visible_frames_inside_aperture": longest_run(inside_aperture_indices),
            "volume_eligible_frames": len(volume_indices),
            "max_consecutive_visible_frames_inside_volume": longest_run(inside_volume_indices),
        }
        seq_out.append(sequence_values)
        values[sid] = sequence_values

    pair_out: list[dict[str, object]] = []
    pair_outcomes: dict[str, Counter[str]] = {name: Counter() for name in [
        "mean_distance_change_per_0_2s_step", "fraction_steps_approaching_center", "aperture_entries_2d",
        "aperture_exits_2d", "max_consecutive_visible_frames_inside_aperture",
        "max_consecutive_visible_frames_inside_volume",
    ]}
    for i in range(1, 21):
        pair_id = f"oracle_pair_{i:03d}"
        pos, neg = values[pair_id + "_POS"], values[pair_id + "_NEG"]
        row: dict[str, object] = {"pair_id": pair_id}
        for metric in pair_outcomes:
            higher_is_better = metric != "mean_distance_change_per_0_2s_step" and metric != "aperture_exits_2d"
            result = pair_result(pos[metric], neg[metric], higher_is_better)
            pair_outcomes[metric][result] += 1
            row[f"positive_{metric}"] = pos[metric]
            row[f"negative_{metric}"] = neg[metric]
            row[f"{metric}_pair_result"] = result
        pair_out.append(row)
    return frame_out, seq_out, pair_out, pair_outcomes


def write_summary(path: Path, seq_rows: list[dict[str, object]], outcomes: dict[str, Counter[str]]) -> None:
    passing = sum(int(row["aperture_eligible_frames"]) >= 2 for row in seq_rows)
    lines = [
        "# Diagnósticos temporais oracle — resumo exploratório v1",
        "",
        "**Escopo:** 20 pares TRAIN congelados, frames 16–26. Protocolo complementar exploratório; sem gate e sem ajuste de modelo.",
        "",
        f"- Sequências com pelo menos um passo consecutivo BALL+abertura avaliável: {passing}/40.",
        "- Cada comparação pareada usa apenas sequências com medida disponível nos dois lados; empate permanece explícito.",
        "- Gaps BALL, frames `PARTIAL` e geometria fora/indisponível quebram a continuidade; nenhuma interpolação foi feita.",
        "",
        "| Medida (favorável ao positivo) | Positivo vence | Empate | Controle vence | Não avaliável |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "mean_distance_change_per_0_2s_step": "Variação média da distância ao centro (menor é aproximação)",
        "fraction_steps_approaching_center": "Fração dos passos aproximando-se do centro",
        "aperture_entries_2d": "Entradas 2D no quadrilátero",
        "aperture_exits_2d": "Saídas 2D do quadrilátero (menor)",
        "max_consecutive_visible_frames_inside_aperture": "Maior persistência 2D na abertura",
        "max_consecutive_visible_frames_inside_volume": "Maior persistência 2D no volume",
    }
    for metric, counter in outcomes.items():
        lines.append(f"| {labels[metric]} | {counter['POSITIVE_WIN']} | {counter['TIE']} | {counter['NEGATIVE_WIN']} | {counter['NOT_EVALUABLE']} |")
    lines.extend([
        "",
        "As comparações são descritivas e posteriores ao resultado espacial v1. Não selecionar uma métrica vencedora, não criar score combinado e não usar este lote para afirmar desempenho geral. Entradas, saídas e persistência são estados em polígonos 2D projetados, não cruzamento físico 3D. Consulte `oracle_temporal_diagnostics_addendum_v1.md` para a regra completa.",
        "",
        "Artefatos: `oracle_temporal_diagnostic_frames_v1.csv`, `oracle_temporal_diagnostic_sequences_v1.csv`, `oracle_temporal_diagnostic_pairs_v1.csv`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frame-output", type=Path, default=DEFAULT_FRAMES_OUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQ_OUT)
    parser.add_argument("--pair-output", type=Path, default=DEFAULT_PAIRS_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    frame_rows, sequence_rows, pair_rows, outcomes = extract(args.input)
    save_csv(args.frame_output, frame_rows)
    save_csv(args.sequence_output, sequence_rows)
    save_csv(args.pair_output, pair_rows)
    write_summary(args.summary, sequence_rows, outcomes)
    for metric, counts in outcomes.items():
        print(metric, dict(counts))
    print(f"Resumo: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
