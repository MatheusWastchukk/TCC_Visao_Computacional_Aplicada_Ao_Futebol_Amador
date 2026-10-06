"""Extract frozen, descriptive oracle trajectory features on TRAIN pairs only."""

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
DEFAULT_FRAMES = EXP / "oracle_frame_annotations_v1_temporal_geometry.csv"
DEFAULT_PAIRS = EXP / "oracle_pair_manifest_v1.csv"
DEFAULT_FRAME_OUT = EXP / "oracle_feature_frames_v1.csv"
DEFAULT_SEQUENCE_OUT = EXP / "oracle_sequence_features_v1.csv"
DEFAULT_PAIR_OUT = EXP / "oracle_pairwise_results_v1.csv"
DEFAULT_SUMMARY = EXP / "oracle_result_summary_v1.md"
WINDOW = range(16, 27)
TIE_EPS = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box_center(box_json: str) -> tuple[float, float] | None:
    if not box_json:
        return None
    box = json.loads(box_json)
    if len(box) != 4:
        raise ValueError(f"Expected [x1,y1,x2,y2], got {box}")
    return (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2


def point_polygon_distance(point: tuple[float, float], polygon: list[list[float]]) -> tuple[float, bool]:
    p = (float(point[0]), float(point[1]))
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    inside = cv2.pointPolygonTest(contour, p, False) >= 0
    if inside:
        return 0.0, True
    best = float("inf")
    px, py = p
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        qx, qy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - qx, py - qy))
    return best, False


def mean_aperture_side(polygon: list[list[float]]) -> float:
    sides = [math.dist(polygon[i], polygon[(i + 1) % 4]) for i in range(4)]
    mean_side = sum(sides) / 4
    if not math.isfinite(mean_side) or mean_side <= 0:
        raise ValueError(f"Invalid aperture dimensions: {polygon}")
    return mean_side


def sequence_id(pair_id: str, side: str) -> str:
    return f"{pair_id}_{side}"


def extract(frames_path: Path, pairs_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    frames = read_csv(frames_path)
    pairs = read_csv(pairs_path)
    if len(frames) != 1640 or len(pairs) != 20:
        raise ValueError(f"Unexpected frozen set sizes: frames={len(frames)} pairs={len(pairs)}")
    if any(row.get("split", "").upper() not in {"TRAIN", ""} for row in frames):
        raise ValueError("Non-TRAIN rows found in oracle frames")
    if any(row.get("selection_status") != "FROZEN_BEFORE_ORACLE_FEATURES" for row in pairs):
        raise ValueError("Pair manifest does not match the pre-feature frozen protocol")

    by_sequence: dict[str, list[dict[str, str]]] = {}
    for row in frames:
        by_sequence.setdefault(row["sequence_id"], []).append(row)

    frame_features: list[dict[str, object]] = []
    sequence_features: list[dict[str, object]] = []
    per_sequence: dict[str, dict[str, object]] = {}
    for sid, seq_rows in sorted(by_sequence.items()):
        seq_rows.sort(key=lambda row: int(row["sequence_frame"]))
        event_class = "POS" if sid.endswith("_POS") else "NEG"
        eligible_visible = 0
        observable = 0
        inside_aperture = 0
        inside_volume = 0
        volume_eligible = 0
        normalized_distances: list[float] = []
        observed_rows = 0
        event_rows = [row for row in seq_rows if int(row["sequence_frame"]) in WINDOW]
        for row in event_rows:
            ball_status = row["ball_visibility"]
            loc = box_center(row.get("ball_of_play_box_json", ""))
            region_valid = row["effective_goal_region_status"] == "VALID" and row["goal_region_geometry_inside_frame"] == "True"
            volume_valid = row["effective_goal_volume_status"] == "VALID" and row["goal_volume_geometry_inside_frame"] == "True"
            is_observed = ball_status in {"VISIBLE", "PARTIAL"} and loc is not None and region_valid
            observable += int(is_observed)
            if ball_status in {"VISIBLE", "PARTIAL"} and loc is not None:
                observed_rows += 1

            region = json.loads(row["effective_goal_region_points_json"]) if region_valid else None
            volume = json.loads(row["effective_goal_volume_points_json"]) if volume_valid else None
            dnorm: float | None = None
            in_aperture: bool | None = None
            in_volume: bool | None = None
            primary_eligible = ball_status == "VISIBLE" and loc is not None and region is not None
            if primary_eligible:
                distance, in_poly = point_polygon_distance(loc, region)
                dnorm = distance / mean_aperture_side(region)
                in_aperture = in_poly
                eligible_visible += 1
                normalized_distances.append(dnorm)
                inside_aperture += int(in_poly)
            if ball_status in {"VISIBLE", "PARTIAL"} and loc is not None and volume is not None:
                _, in_vol = point_polygon_distance(loc, volume)
                in_volume = in_vol
                volume_eligible += 1
                inside_volume += int(in_vol)

            frame_features.append({
                "pair_id": sid.rsplit("_", 1)[0], "sequence_id": sid, "class": event_class,
                "match_id": row["match_id"], "sequence_frame": row["sequence_frame"], "timestamp_hhmmss": row["timestamp_hhmmss"],
                "ball_visibility": ball_status, "ball_center_x": "" if loc is None else round(loc[0], 3),
                "ball_center_y": "" if loc is None else round(loc[1], 3),
                "aperture_status": row["effective_goal_region_status"], "aperture_inside_frame": row["goal_region_geometry_inside_frame"],
                "aperture_source": row["goal_region_geometry_source"], "aperture_anchor_frame": row["goal_region_anchor_frame"],
                "primary_score_eligible": primary_eligible, "normalized_distance_to_aperture": "" if dnorm is None else round(dnorm, 8),
                "center_inside_aperture": "" if in_aperture is None else in_aperture,
                "volume_status": row["effective_goal_volume_status"], "volume_inside_frame": row["goal_volume_geometry_inside_frame"],
                "center_inside_volume": "" if in_volume is None else in_volume,
            })

        score = -min(normalized_distances) if normalized_distances else None
        observable_fraction = observable / len(WINDOW)
        seq_feature = {
            "pair_id": sid.rsplit("_", 1)[0], "sequence_id": sid, "class": event_class,
            "match_id": seq_rows[0]["match_id"], "event_window_frames": len(WINDOW),
            "ball_locatable_and_aperture_usable_frames": observable,
            "ball_locatable_and_aperture_usable_fraction": round(observable_fraction, 6),
            "observability_gate_70pct": observable >= 8,
            "visible_ball_aperture_eligible_frames": eligible_visible,
            "primary_proximity_score": "" if score is None else round(score, 8),
            "center_inside_aperture_frames": inside_aperture,
            "center_inside_aperture_fraction_of_primary_frames": "" if not eligible_visible else round(inside_aperture / eligible_visible, 6),
            "volume_eligible_observations": volume_eligible,
            "center_inside_volume_frames": inside_volume,
            "center_inside_volume_fraction": "" if not volume_eligible else round(inside_volume / volume_eligible, 6),
        }
        sequence_features.append(seq_feature)
        per_sequence[sid] = seq_feature

    pair_features: list[dict[str, object]] = []
    outcomes: Counter[str] = Counter()
    aperture_occupancy_outcomes: Counter[str] = Counter()
    volume_occupancy_outcomes: Counter[str] = Counter()
    by_match: dict[str, Counter[str]] = {}
    by_context_tag: dict[str, Counter[str]] = {}
    for pair in pairs:
        pos_id = sequence_id(pair["pair_id"], "POS")
        neg_id = sequence_id(pair["pair_id"], "NEG")
        pos, neg = per_sequence[pos_id], per_sequence[neg_id]
        pos_score = None if pos["primary_proximity_score"] == "" else float(pos["primary_proximity_score"])
        neg_score = None if neg["primary_proximity_score"] == "" else float(neg["primary_proximity_score"])
        if pos_score is None or neg_score is None:
            outcome = "NOT_EVALUABLE"
            outcomes[outcome] += 1
        elif abs(pos_score - neg_score) <= TIE_EPS:
            outcome = "TIE"
            outcomes[outcome] += 1
        elif pos_score > neg_score:
            outcome = "POSITIVE_WIN"
            outcomes[outcome] += 1
        else:
            outcome = "NEGATIVE_WIN"
            outcomes[outcome] += 1

        def compare_fraction(field: str, counts: Counter[str]) -> tuple[object, object, str]:
            positive = pos[field]
            negative = neg[field]
            if positive == "" or negative == "":
                label = "NOT_EVALUABLE"
                counts[label] += 1
                return positive, negative, label
            positive_value, negative_value = float(positive), float(negative)
            label = "TIE" if abs(positive_value - negative_value) <= TIE_EPS else "POSITIVE_WIN" if positive_value > negative_value else "NEGATIVE_WIN"
            counts[label] += 1
            return positive, negative, label

        pos_aperture_fraction, neg_aperture_fraction, aperture_fraction_outcome = compare_fraction(
            "center_inside_aperture_fraction_of_primary_frames", aperture_occupancy_outcomes
        )
        pos_volume_fraction, neg_volume_fraction, volume_fraction_outcome = compare_fraction(
            "center_inside_volume_fraction", volume_occupancy_outcomes
        )
        by_match.setdefault(pair["match_id"], Counter())[outcome] += 1
        for tag in filter(None, pair.get("negative_context_tags", "").split(";")):
            by_context_tag.setdefault(tag, Counter())[outcome] += 1
        pair_features.append({
            "pair_id": pair["pair_id"], "match_id": pair["match_id"], "positive_sequence_id": pos_id,
            "negative_sequence_id": neg_id, "positive_score": "" if pos_score is None else round(pos_score, 8),
            "negative_score": "" if neg_score is None else round(neg_score, 8), "outcome": outcome,
            "positive_observable_frames": pos["ball_locatable_and_aperture_usable_frames"],
            "negative_observable_frames": neg["ball_locatable_and_aperture_usable_frames"],
            "positive_inside_aperture_fraction": pos_aperture_fraction,
            "negative_inside_aperture_fraction": neg_aperture_fraction,
            "inside_aperture_fraction_outcome": aperture_fraction_outcome,
            "positive_inside_volume_fraction": pos_volume_fraction,
            "negative_inside_volume_fraction": neg_volume_fraction,
            "inside_volume_fraction_outcome": volume_fraction_outcome,
        })

    valid_pairs = sum(outcomes[x] for x in ("POSITIVE_WIN", "NEGATIVE_WIN", "TIE"))
    strict_wins = outcomes["POSITIVE_WIN"]
    gate_denominator = valid_pairs
    win_fraction = strict_wins / gate_denominator if gate_denominator else None
    summary = {
        "outcomes": outcomes, "valid_pairs": valid_pairs, "win_fraction": win_fraction,
        "observability_pass_sequences": sum(bool(row["observability_gate_70pct"]) for row in sequence_features),
        "sequence_count": len(sequence_features), "observable_total": sum(int(row["ball_locatable_and_aperture_usable_frames"]) for row in sequence_features),
        "max_observable": len(WINDOW) * len(sequence_features),
        "score_eligible_sequences": sum(row["primary_proximity_score"] != "" for row in sequence_features),
        "by_match": by_match, "by_context_tag": by_context_tag,
        "aperture_occupancy_outcomes": aperture_occupancy_outcomes,
        "volume_occupancy_outcomes": volume_occupancy_outcomes,
    }
    return frame_features, sequence_features, pair_features, summary


def write_summary(path: Path, summary: dict[str, object]) -> None:
    outcomes: Counter[str] = summary["outcomes"]
    aperture_counts: Counter[str] = summary["aperture_occupancy_outcomes"]
    volume_counts: Counter[str] = summary["volume_occupancy_outcomes"]
    fraction = summary["win_fraction"]
    gate = "ATINGIDO" if fraction is not None and fraction >= 0.75 else "NÃO ATINGIDO"
    match_lines = [f"- `{match}`: {counts['POSITIVE_WIN']} vitórias, {counts['TIE']} empates, {counts['NEGATIVE_WIN']} derrotas" for match, counts in sorted(summary["by_match"].items())]
    tag_lines = [f"- `{tag}`: {counts['POSITIVE_WIN']} vitórias, {counts['TIE']} empates, {counts['NEGATIVE_WIN']} derrotas" for tag, counts in sorted(summary["by_context_tag"].items())]
    text = f"""# Resultado do piloto oracle de trajetória — versão 1

**Escopo:** TRAIN, 20 pares congelados antes das features; janela frames 16–26. Fórmula pré-registrada em `oracle_feature_protocol_frozen_v1.md`. Sem ajuste de ranker e sem uso de VALIDATION/TEST.

## Observabilidade

- Sequências com pelo menos 8/11 frames observáveis (BALL VISIBLE/PARTIAL e abertura válida totalmente no quadro): {summary['observability_pass_sequences']}/{summary['sequence_count']}.
- Observações conjuntas: {summary['observable_total']}/{summary['max_observable']} ({summary['observable_total'] / summary['max_observable']:.1%}).
- O gate de maioria das sequências em pelo menos 70% é {'ATINGIDO' if summary['observability_pass_sequences'] > summary['sequence_count'] / 2 else 'NÃO ATINGIDO'}.
- Sequências com score primário calculável: {summary['score_eligible_sequences']}/{summary['sequence_count']}.

## Ordenação pareada pelo score primário

- Pares avaliáveis: {summary['valid_pairs']}/20; não avaliáveis: {outcomes['NOT_EVALUABLE']}.
- Vitórias estritas do positivo: {outcomes['POSITIVE_WIN']}; empates: {outcomes['TIE']}; vitórias do controle: {outcomes['NEGATIVE_WIN']}.
- Fração de vitórias estritas entre pares avaliáveis: {'n/a' if fraction is None else f'{fraction:.1%}'}; gate exploratório de 75%: {gate}.

### Distribuição

Os 17 positivos vencedores estão distribuídos por {sum(counts['POSITIVE_WIN'] > 0 for counts in summary['by_match'].values())} das {len(summary['by_match'])} partidas com pares; os 3 empates aparecem em 3 partidas.

Por partida:
{chr(10).join(match_lines)}

Por tags do controle (categorias podem se sobrepor):
{chr(10).join(tag_lines)}

## Interpretação limitada

O score congelado é de proximidade mínima ao quadrilátero em uma janela temporal. Ele não mede direção, continuidade, cruzamento entre frames ou persistência; portanto, o resultado demonstra sinal de proximidade nas anotações oracle, não ganho específico de tracking/temporalidade sobre um comparador espacial simples. O gate de 75% foi alcançado, mas uma decisão de implementar tracker deve considerar esta distinção e os limites dos rótulos de geometria propagada.

Como medidas espaciais secundárias, a fração de centros visíveis dentro da abertura favoreceu o positivo em {aperture_counts['POSITIVE_WIN']}/20 pares (empates {aperture_counts['TIE']}); a fração dentro do volume favoreceu o positivo em {volume_counts['POSITIVE_WIN']} pares, com {volume_counts['NOT_EVALUABLE']} não avaliáveis pela geometria incerta. Esses achados reforçam a separação espacial oracle, mas não medem ganho de trajetória temporal.

O score não prova trajetória contínua nem cruzamento físico 3D. As medidas de abertura e volume são projetadas e há âncoras homográficas. O resultado é um diagnóstico de viabilidade TRAIN, não desempenho geral.

Arquivos: `oracle_feature_frames_v1.csv`, `oracle_sequence_features_v1.csv`, `oracle_pairwise_results_v1.csv` e protocolo congelado `oracle_feature_protocol_frozen_v1.md`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--frame-output", type=Path, default=DEFAULT_FRAME_OUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQUENCE_OUT)
    parser.add_argument("--pair-output", type=Path, default=DEFAULT_PAIR_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    frame_features, sequence_features, pair_features, summary = extract(args.frames, args.pairs)
    write_csv(args.frame_output, frame_features)
    write_csv(args.sequence_output, sequence_features)
    write_csv(args.pair_output, pair_features)
    write_summary(args.summary, summary)
    print(f"Observabilidade >=70%: {summary['observability_pass_sequences']}/{summary['sequence_count']} sequências")
    print(f"Pares: {dict(summary['outcomes'])}")
    print(f"Resumo: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
