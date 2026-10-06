"""Summarize the frozen tracker run against oracle BALL annotations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parent
RUN = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "run_v1"
INPUT = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "oracle_tracker_input_frames_16_26.csv"
DEFAULT_SEQ = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "per_sequence_v1.csv"
DEFAULT_STATUS = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "by_visibility_v1.csv"
DEFAULT_SUMMARY = EXP / "ORACLE-TRACKER-BASELINE-EXP-0001" / "comparison_summary_v1.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    targets = [r for r in rows if r["target_annotated"] == "True"]
    return {
        "frames": len(rows), "target_frames": len(targets),
        "candidate_frames": sum(r["target_candidate_rank"] != "0" for r in targets),
        "identity_correct_frames": sum(r["identity_correct"] == "True" for r in targets),
        "wrong_switch_frames": sum(r["wrong_ball_switch"] == "True" for r in targets),
        "candidate_recall": sum(r["target_candidate_rank"] != "0" for r in targets) / len(targets) if targets else None,
        "identity_accuracy": sum(r["identity_correct"] == "True" for r in targets) / len(targets) if targets else None,
        "wrong_switch_rate": sum(r["wrong_ball_switch"] == "True" for r in targets) / len(targets) if targets else None,
    }


def extract(run_path: Path, input_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    result_rows = read_csv(run_path / "frame_identity_results.csv")
    input_rows = read_csv(input_path)
    metadata = {(r["sequence_id"], r["sequence_frame"]): r for r in input_rows}
    if len(result_rows) != 440 or len(metadata) != 440:
        raise ValueError(f"Expected 440 tracker/input rows, got {len(result_rows)}/{len(metadata)}")
    joined = []
    for result in result_rows:
        source = metadata[(result["sequence_id"], result["sequence_frame"])]
        joined.append({**result, **{k: source[k] for k in ("match_id", "ball_visibility", "distractor_boxes_json", "ball_of_play_box_json")}})

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in joined:
        groups[row["sequence_id"]].append(row)
    by_sequence: list[dict[str, object]] = []
    for sid, rows in sorted(groups.items()):
        info = metrics(rows)
        by_sequence.append({
            "sequence_id": sid, "pair_id": sid.rsplit("_", 1)[0], "class": sid.rsplit("_", 1)[1],
            "match_id": rows[0]["match_id"], **info,
        })

    by_visibility: list[dict[str, object]] = []
    for key, subset in [
        ("ALL_TARGET", [r for r in joined if r["target_annotated"] == "True"]),
        ("VISIBLE", [r for r in joined if r["ball_visibility"] == "VISIBLE" and r["target_annotated"] == "True"]),
        ("PARTIAL", [r for r in joined if r["ball_visibility"] == "PARTIAL" and r["target_annotated"] == "True"]),
        ("POSITIVE", [r for r in joined if r["sequence_id"].endswith("_POS") and r["target_annotated"] == "True"]),
        ("NEGATIVE", [r for r in joined if r["sequence_id"].endswith("_NEG") and r["target_annotated"] == "True"]),
    ]:
        by_visibility.append({"group": key, **metrics(subset)})

    status_counts = Counter(r["ball_visibility"] for r in joined)
    distractor_annotated_frames = sum(bool(json.loads(r.get("distractor_boxes_json", "[]") or "[]")) for r in joined)
    size_summary: dict[str, dict[str, object]] = {}
    for label in ("POS", "NEG"):
        for candidate_present in (True, False):
            subset = [
                r for r in joined
                if r["sequence_id"].endswith("_" + label)
                and r["target_annotated"] == "True"
                and (r["target_candidate_rank"] != "0") == candidate_present
            ]
            widths, heights, areas = [], [], []
            for row in subset:
                x1, y1, x2, y2 = map(float, json.loads(row["ball_of_play_box_json"]))
                widths.append(x2 - x1)
                heights.append(y2 - y1)
                areas.append((x2 - x1) * (y2 - y1))
            size_summary[f"{label}_{'candidate' if candidate_present else 'miss'}"] = {
                "count": len(subset), "median_width": statistics.median(widths) if widths else None,
                "median_height": statistics.median(heights) if heights else None,
                "median_area": statistics.median(areas) if areas else None,
            }
    total = metrics(joined)
    summary = {
        "metrics": total,
        "visibility_counts": status_counts,
        "distractor_annotated_frames": distractor_annotated_frames,
        "sequences": len(groups),
        "positive_sequences": sum(sid.endswith("_POS") for sid in groups),
        "negative_sequences": sum(sid.endswith("_NEG") for sid in groups),
        "positive_candidate_recall": next(r["candidate_recall"] for r in by_visibility if r["group"] == "POSITIVE"),
        "negative_candidate_recall": next(r["candidate_recall"] for r in by_visibility if r["group"] == "NEGATIVE"),
        "box_size": size_summary,
    }
    return by_sequence, by_visibility, summary


def write_summary(path: Path, by_sequence: list[dict[str, object]], by_visibility: list[dict[str, object]], summary: dict[str, object]) -> None:
    m = summary["metrics"]
    lines = [
        "# Comparação do tracker existente com o oracle — v1",
        "",
        "**Escopo:** 40 sequências TRAIN, frames 16–26; modelo BALL-EXP-0005, Viterbi existente, conf=0.01, max_candidates=10, imgsz=960, CPU. Protocolo em `oracle_tracker_comparison_protocol_v1.md`.",
        "",
        "## Resultado agregado",
        "",
        f"- Sequências/imagens: {summary['sequences']}/440; alvos BALL `VISIBLE/PARTIAL`: {m['target_frames']}.",
        f"- Candidate recall (alguma proposta com IoU ≥0,50): {m['candidate_frames']}/{m['target_frames']} ({m['candidate_recall']:.1%}).",
        f"- Identity accuracy (proposta selecionada com IoU ≥0,50): {m['identity_correct_frames']}/{m['target_frames']} ({m['identity_accuracy']:.1%}).",
        f"- Frames-alvo com distractor anotado: {summary['distractor_annotated_frames']}/{m['target_frames']}; trocas marcadas: {m['wrong_switch_frames']}. Uma taxa zero sem cobertura adequada de distractors não prova ausência de troca.",
        "",
        "## Quebra por visibilidade e classe",
        "",
        "| Grupo | Alvos | Candidate recall | Identity accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in by_visibility:
        lines.append(f"| {row['group']} | {row['target_frames']} | {row['candidate_recall']:.1%} | {row['identity_accuracy']:.1%} |")
    lines.extend([
        "",
        "## Tamanho das caixas humanas de BALL (descrição pós-hoc)",
        "",
        "| Classe e resultado do detector | Frames | Mediana largura × altura (px) | Mediana área (px²) |",
        "|---|---:|---:|---:|",
    ])
    for key in ("POS_candidate", "POS_miss", "NEG_candidate", "NEG_miss"):
        item = summary["box_size"][key]
        lines.append(f"| {key} | {item['count']} | {item['median_width']:.1f} × {item['median_height']:.1f} | {item['median_area']:.1f} |")
    lines.extend([
        "",
        "## Limites e decisão operacional",
        "",
        "O detector deixou de propor qualquer caixa compatível em 56,0% dos alvos anotados; isso limita o teto do tracker independentemente da associação. A seleção Viterbi ficou correta em 30,6% dos alvos. Este é um diagnóstico nos clipes oracle selecionados, não métrica geral. Geometria temporal da goleira continua estimada por âncoras/homografias.",
        "",
        "**Decisão:** NO-GO para integrar o tracker atual ou suas features no V4. Manter somente como referência experimental. Para continuar a linha, primeiro é necessário explicar/reduzir a baixa candidate recall com evidência nos casos anotados; não retunar thresholds neste lote após ver o resultado. Baseline congelado; VALIDATION/TEST fechados.",
        "",
        "Uma primeira inspeção visual está em `detector_miss_review_v1/contact_sheet_*.jpg`; a fila `positive_detector_misses_v1.csv` deixa uma taxonomia de causas em branco, pois estas figuras servem para revisar, não para inferir rótulos sem validação. Artefatos: `run_v1/metrics.json`, `per_sequence_v1.csv`, `by_visibility_v1.csv`, `run_v1/frame_identity_results.csv`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQ)
    parser.add_argument("--visibility-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    sequence_rows, visibility_rows, summary = extract(args.run, args.input)
    write_csv(args.sequence_output, sequence_rows)
    write_csv(args.visibility_output, visibility_rows)
    write_summary(args.summary, sequence_rows, visibility_rows, summary)
    print(json.dumps({"metrics": summary["metrics"], "visibility": dict(summary["visibility_counts"]), "distractor_annotated_frames": summary["distractor_annotated_frames"]}, ensure_ascii=False))
    print(f"Resumo: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
