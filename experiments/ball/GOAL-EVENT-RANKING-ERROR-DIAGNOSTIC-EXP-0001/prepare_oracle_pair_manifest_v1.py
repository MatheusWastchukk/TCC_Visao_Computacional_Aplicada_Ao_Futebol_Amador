"""Congela os 20 pares humanos existentes para o piloto oracle TRAIN."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
TAXONOMY = EXP / "control_context_taxonomy_v1.csv"
PAIR_SOURCE = EXP / "pairwise_control_audit_v1_annotated.csv"
PRIOR_TRACKS = ROOT / "experiments/ball/BALL-EVENT-TRAJECTORY-DATASET-V1/event_trajectory_sequences_v1.csv"
VIDEOS = ROOT / "data/Videos_Edited"
WINDOW_SECONDS = 8.0
SAMPLE_FPS = 5.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_info(path: Path) -> tuple[float, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Nao foi possivel abrir o video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"Metadados invalidos no video {path}: fps={fps}, frames={count}")
    return count / fps, fps, width, height


def existing_manual_trajectory() -> list[dict[str, str]]:
    if not PRIOR_TRACKS.is_file():
        return []
    rows = read_csv(PRIOR_TRACKS)
    return [row for row in rows if row.get("annotation_status") == "completed"
            and row.get("annotator") == "manual"
            and row.get("ball_visibility") in {"VISIBLE", "PARTIAL", "NOT_VISIBLE", "UNCERTAIN"}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXP / "oracle_pair_manifest_v1.csv")
    parser.add_argument("--summary", type=Path, default=EXP / "oracle_pair_selection_v1.md")
    args = parser.parse_args()

    pairs = read_csv(PAIR_SOURCE)
    taxonomy = {row["control_group_uid"]: row for row in read_csv(TAXONOMY)}
    if len(pairs) != 20 or len(taxonomy) != 20:
        raise ValueError(f"Esperados 20 controles; pares={len(pairs)}, taxonomia={len(taxonomy)}")

    tracks = existing_manual_trajectory()
    duration_cache: dict[str, tuple[float, float, int, int]] = {}
    output_rows: list[dict[str, str]] = []
    seen_goals: set[str] = set()
    seen_controls: set[str] = set()
    for index, pair in enumerate(pairs, start=1):
        control_uid = pair["control_group_uid"]
        annotated = taxonomy.get(control_uid)
        if pair.get("manual_label") != "NO_GOAL" or pair.get("manual_status") != "COMPLETED":
            raise ValueError(f"Controle nao confirmado: {control_uid}")
        if not annotated or annotated.get("taxonomy_status") != "COMPLETED":
            raise ValueError(f"Taxonomia pendente: {control_uid}")
        goal_id = pair["goal_catalog_id"]
        if goal_id in seen_goals or control_uid in seen_controls:
            raise ValueError(f"Par duplicado: goal={goal_id}, control={control_uid}")
        seen_goals.add(goal_id)
        seen_controls.add(control_uid)

        video_name = pair["video_filename"]
        if video_name not in duration_cache:
            duration_cache[video_name] = video_info(VIDEOS / video_name)
        duration, fps, width, height = duration_cache[video_name]
        positive_t = float(pair["goal_timestamp_seconds"])
        negative_t = float(pair["control_candidate_timestamp_seconds"])
        half = WINDOW_SECONDS / 2.0
        pos_start, pos_end = max(0.0, positive_t - half), min(duration, positive_t + half)
        neg_start, neg_end = max(0.0, negative_t - half), min(duration, negative_t + half)
        nearby = [row for row in tracks if row.get("match_id") == pair["match_id"]
                  and abs(float(row["timestamp_seconds"]) - positive_t) <= half]
        nearby.sort(key=lambda row: float(row["timestamp_seconds"]))
        control_nearby = [row for row in tracks if row.get("match_id") == pair["match_id"]
                          and abs(float(row["timestamp_seconds"]) - negative_t) <= half]
        context_tags = annotated.get("context_tags", "")
        observations = annotated.get("direct_observations", "")
        row = {
            "pair_id": f"oracle_pair_{index:03d}",
            "match_id": pair["match_id"],
            "video_filename": video_name,
            "positive_catalog_goal_id": goal_id,
            "positive_timestamp_seconds": f"{positive_t:.3f}",
            "positive_window_start_seconds": f"{pos_start:.3f}",
            "positive_window_end_seconds": f"{pos_end:.3f}",
            "positive_label_source": "CATALOGED_GOAL_TIMESTAMP",
            "positive_annotation_status": "PARTIAL_EXISTING_MANUAL" if nearby else "PENDING_MANUAL",
            "positive_existing_manual_frames": str(len(nearby)),
            "positive_existing_manual_time_start": f"{float(nearby[0]['timestamp_seconds']):.3f}" if nearby else "",
            "positive_existing_manual_time_end": f"{float(nearby[-1]['timestamp_seconds']):.3f}" if nearby else "",
            "positive_existing_manual_source": "experiments/ball/BALL-EVENT-TRAJECTORY-DATASET-V1/event_trajectory_sequences_v1.csv" if nearby else "",
            "negative_control_group_uid": control_uid,
            "negative_timestamp_seconds": f"{negative_t:.3f}",
            "negative_window_start_seconds": f"{neg_start:.3f}",
            "negative_window_end_seconds": f"{neg_end:.3f}",
            "negative_label_source": "USER_CONFIRMED_NO_GOAL",
            "negative_context_tags": context_tags,
            "negative_direct_observations": observations,
            "negative_annotation_status": "PENDING_MANUAL",
            "negative_existing_manual_frames": str(len(control_nearby)),
            "video_duration_seconds": f"{duration:.3f}",
            "video_fps": f"{fps:.6f}",
            "frame_width": str(width),
            "frame_height": str(height),
            "window_seconds_requested": f"{WINDOW_SECONDS:.1f}",
            "sample_fps_target": f"{SAMPLE_FPS:.1f}",
            "selection_status": "FROZEN_BEFORE_ORACLE_FEATURES",
        }
        output_rows.append(row)

    fields = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    matches = Counter(row["match_id"] for row in output_rows)
    contexts = Counter(tag for row in output_rows for tag in row["negative_context_tags"].split(";") if tag)
    reused_pos_frames = sum(int(row["positive_existing_manual_frames"]) for row in output_rows)
    reused_neg_frames = sum(int(row["negative_existing_manual_frames"]) for row in output_rows)
    source_hashes = {str(PAIR_SOURCE.relative_to(ROOT)): sha256(PAIR_SOURCE),
                     str(TAXONOMY.relative_to(ROOT)): sha256(TAXONOMY),
                     str(PRIOR_TRACKS.relative_to(ROOT)): sha256(PRIOR_TRACKS)}
    summary = [
        "# Seleção congelada do piloto oracle — versão 1",
        "",
        "**Split:** somente TRAIN. **Features/modelos:** nenhum calculado nesta preparação. **TEST:** fechado.",
        "",
        "## Regra de seleção",
        "",
        "Usar todos os 20 controles já confirmados pelo usuário como NO_GOAL e já categorizados. Cada controle permanece pareado ao timestamp de gol catalogado associado na auditoria de ranking; o par é da mesma partida. Não selecionar/excluir casos após inspecionar features oracle.",
        "",
        f"Foram congelados {len(output_rows)} pares únicos em {len(matches)} partidas ({sum(matches.values())} positivos por timestamp e {sum(matches.values())} controles negativos confirmados). Cada segmento tem janela-alvo de {WINDOW_SECONDS:.0f} s centrada no timestamp; limites são cortados somente nas bordas reais do vídeo. Amostragem de referência: {SAMPLE_FPS:.0f} FPS.",
        "",
        "## Controles: contexto registrado pelo usuário",
        "",
        "| Tag | Casos |",
        "|---|---:|",
    ]
    summary.extend(f"| `{tag}` | {count} |" for tag, count in sorted(contexts.items()))
    summary.extend([
        "",
        "As tags podem coexistir. `SECONDARY_GOAL_VISIVEL` significa que o usuário registrou uma estrutura secundária visível no segmento; não afirma que o candidato se associava a ela nem que causou o score alto.",
        "",
        "## Reuso de anotação de trajetória já existente",
        "",
        f"A busca de sobreposição exata encontrou {reused_pos_frames} frames manuais positivos e {reused_neg_frames} frames manuais negativos no dataset de trajetória existente. O reuso positivo é parcial e não inclui anotação completa da abertura para todos os frames; conferir no manifesto quais timestamps e fonte foram encontrados. Nenhum negativo deste lote já tinha trajetória manual correspondente.",
        "",
        "Isso evita duplicar anotações disponíveis, mas não muda o número de pares nem permite afirmar generalização. A seleção ficou congelada antes do cálculo de qualquer feature oracle.",
        "",
        "## Distribuição por partida",
        "",
        "| Partida | Pares |",
        "|---|---:|",
    ])
    summary.extend(f"| `{match}` | {count} |" for match, count in sorted(matches.items()))
    summary.extend([
        "",
        "## Integridade da seleção",
        "",
        "```json",
        json.dumps(source_hashes, ensure_ascii=False, indent=2),
        "```",
        "",
        f"Manifesto: `{args.output.relative_to(ROOT).as_posix()}`.",
    ])
    args.summary.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Pares congelados: {len(output_rows)} em {len(matches)} partidas")
    print(f"Trajetorias manuais reaproveitaveis: positivos={reused_pos_frames} frames; negativos={reused_neg_frames} frames")
    print(f"Manifesto: {args.output}")
    print(f"Resumo: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
