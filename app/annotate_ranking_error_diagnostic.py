"""Revisa clips TRAIN com caixas BALL/GOAL historicamente persistidas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
LABEL_KEYS = {"g": "GOAL", "n": "NO_GOAL", "u": "NOT_EVALUABLE"}
CAUSES = {
    "1": "B0_SEM_PROPOSTA_UTIL", "2": "B1_CAIXA_IMPRECISA", "3": "B2_MULTIPLICIDADE",
    "4": "B3_CONTINUIDADE_VISUAL", "5": "G1_GOLEIRA_SECUNDARIA", "6": "G2_ASSOCIACAO_GOLEIRA",
    "7": "R1_FALTA_SINAL_NAS_FEATURES", "8": "R2_SINAL_PRESENTE_MAS_SCORE_BAIXO",
    "9": "S1_SUPRESSAO", "0": "S2_CORTE_TOPK", "x": "V1_NAO_AVALIAVEL", "o": "OUTRO",
}
VALUES = ("PENDENTE", "SIM", "NAO", "INCERTO")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def save_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def merge_review(source: list[dict[str, str]], fields: list[str], output: Path) -> tuple[list[dict[str, str]], list[str]]:
    added = [
        "manual_goal_label", "manual_status", "ball_visible", "ball_proposal_quality",
        "goal_structure_visible", "candidate_goal_association", "offensive_context",
        "camera_motion", "occlusion_or_aerial", "candidate_secondary_goal",
        "cause_tags", "diagnosis_confidence", "direct_observations", "causal_hypothesis",
    ]
    for field in added:
        if field not in fields:
            fields.append(field)
    previous: dict[str, dict[str, str]] = {}
    if output.is_file():
        old, _ = read_csv(output)
        previous = {row["case_id"]: row for row in old}
    rows = []
    for item in source:
        old = previous.get(item["case_id"], {})
        row = dict(item)
        for field in added:
            if field in old:
                row[field] = old[field]
            else:
                row.setdefault(field, "")
        row["manual_status"] = row.get("manual_status") or "PENDING"
        row["candidate_secondary_goal"] = row.get("candidate_secondary_goal") or "PENDENTE"
        rows.append(row)
    return rows, fields


def cycle(value: str) -> str:
    try:
        return VALUES[(VALUES.index(value) + 1) % len(VALUES)]
    except ValueError:
        return "SIM"


def parse_box(value: str) -> tuple[int, int, int, int] | None:
    try:
        x1, y1, x2, y2 = (float(item) for item in json.loads(value))
        return round(x1), round(y1), round(x2), round(y2)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def detections_by_frame(path: Path) -> dict[int, list[dict[str, str]]]:
    if not path.is_file():
        return {}
    rows, _ = read_csv(path)
    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            by_frame.setdefault(int(row["frame_index"]), []).append(row)
        except (KeyError, ValueError):
            continue
    return by_frame


def draw_box(image, value: str, color: tuple[int, int, int], label: str) -> None:
    box = parse_box(value)
    if not box:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def overlay(image, row: dict[str, str], index: int, total: int, seconds: float, frame_index: int,
            detected: list[dict[str, str]] | None, playing: bool, max_width: int):
    state = "RODANDO" if playing else "PAUSADO"
    height, width = image.shape[:2]
    panel_height = min(210, max(150, height // 5))
    cv2.rectangle(image, (0, 0), (width, panel_height), (0, 0, 0), -1)
    lines = [
        f"{index}/{total} | {row['match_id']} | rank RF {row.get('temporal_rank_full_pool','')} | shortlist {row.get('top60_rank','-')} | {state}",
        f"Grupo: {row['group_uid']} | tempo: {seconds:.1f}s | frame: {frame_index} | score: {row.get('temporal_score_lomo','')}",
        f"Gols catalogados dentro da janela: {row.get('catalog_goals_covered') or 'nenhum'} | {row.get('catalog_goal_timestamps_covered') or ''}",
        f"Rótulo manual: {row.get('manual_goal_label') or 'PENDENTE'} | secundária neste candidato: {row.get('candidate_secondary_goal','PENDENTE')}",
        "GOAL: caixa verde | BALL: caixa magenta | não há IDs de tracker neste artefato",
        "CSV histórico registra apenas frames de propostas filtradas; sem linha não prova ausência de detecção.",
        "SPACE play/pause | = speed 1/2/4x | A/D +/-1s | G/N/U label+next | F restart full clip | Q save/quit",
        "B bola | P qualidade proposta | E estrutura gol | R associação | W ação ofensiva | V câmera | Y oclusão/aérea",
        "Optional: BALL/GOAL details, secondary-goal tags, causes, notes and confidence",
    ]
    for index_line, line in enumerate(lines):
        cv2.putText(image, line[:190], (10, 20 + index_line * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 240, 245), 1, cv2.LINE_AA)
    if detected:
        # candidate_frames pode ter várias propostas por frame; exibir todas, deduplicando caixas repetidas.
        seen: set[tuple[str, str]] = set()
        for proposal in detected:
            goal_key = ("GOAL", proposal.get("goal_box_json", ""))
            if goal_key not in seen:
                seen.add(goal_key)
                draw_box(image, goal_key[1], (0, 255, 0), f"GOAL {proposal.get('goal_confidence','')}")
            ball_key = ("BALL", proposal.get("ball_box_json", ""))
            if ball_key not in seen:
                seen.add(ball_key)
                draw_box(image, ball_key[1], (255, 0, 255), f"BALL {proposal.get('ball_confidence','')}")
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def cycle_observation(row: dict[str, str], field: str,
                      options: tuple[str, ...] = VALUES) -> None:
    try:
        row[field] = options[(options.index(row.get(field, "PENDENTE")) + 1) % len(options)]
    except ValueError:
        row[field] = options[0]


def annotate(rows: list[dict[str, str]], fields: list[str], videos_dir: Path, output: Path,
             secondary_rows: list[dict[str, str]], secondary_fields: list[str], secondary_output: Path,
             max_width: int, match_fps: dict[str, float]) -> int:
    pending_indices = [i for i, row in enumerate(rows) if row.get("manual_status") != "COMPLETED"]
    if not pending_indices:
        print(f"Não há casos pendentes. Anotações: {output}")
        return 0
    win = "Auditoria TRAIN de ranking com caixas BALL/GOAL"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cursor = 0
        while cursor < len(pending_indices):
            row_index = pending_indices[cursor]
            row = rows[row_index]
            video_path = videos_dir / row["video_filename"]
            frames_path = ROOT / row.get("candidate_frames_csv", "")
            frame_detections = detections_by_frame(frames_path)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Vídeo TRAIN não encontrado: {video_path}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or match_fps.get(row["match_id"], 30.0))
                frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
                full_start_frame = max(0, int(float(row["delivery_start_seconds"]) * fps))
                full_end_frame = min(frame_count - 1, int(float(row["delivery_end_seconds"]) * fps))
                # Revisão rápida começa perto do timestamp do candidato; F abre a janela completa de 60 s.
                # Abrir sempre o segmento completo 20+40 s do manifesto.
                start_frame, end_frame = full_start_frame, full_end_frame
                current = start_frame
                playing = False
                playback_rate = 2.0
                seek = True
                while True:
                    current = min(end_frame, max(start_frame, current))
                    if seek:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, current)
                    ok, image = capture.read()
                    if not ok or image is None:
                        playing = False
                        seek = True
                        continue
                    seek = False
                    detected = frame_detections.get(current)
                    image = overlay(image, row, cursor + 1, len(pending_indices), current / fps, current,
                                    detected, playing, max_width)
                    cv2.imshow(win, image)
                    key = cv2.waitKeyEx(max(1, round(1000 / (fps * playback_rate))) if playing else 0)
                    if key in {-1, 255}:
                        if playing:
                            current += max(1, round(playback_rate))
                            if current >= end_frame:
                                current, playing, seek = end_frame, False, True
                        continue
                    char = chr(key & 0xFF).lower()
                    if key == 32 or char == " ":
                        playing = not playing
                        seek = not playing
                    elif char == "f":
                        current, playing, seek = start_frame, False, True
                    elif char == "=":
                        playback_rate = {1.0: 2.0, 2.0: 4.0, 4.0: 1.0}[playback_rate]
                    elif char == "q" or key == 27:
                        save_csv(output, rows, fields)
                        save_csv(secondary_output, secondary_rows, secondary_fields)
                        print(f"Progresso salvo. Retome pelo mesmo comando: {output}")
                        return 0
                    elif char in LABEL_KEYS:
                        row["manual_goal_label"] = LABEL_KEYS[char]
                        row["manual_status"] = "COMPLETED"
                        save_csv(output, rows, fields)
                        break
                    elif char == "s":
                        row["candidate_secondary_goal"] = cycle(row.get("candidate_secondary_goal", "PENDENTE"))
                        target = next((item for item in secondary_rows if item.get("annotation_level") == "CANDIDATE" and item.get("group_uid") == row["group_uid"]), None)
                        if target:
                            target["secondary_goal_status"] = row["candidate_secondary_goal"]
                        save_csv(output, rows, fields)
                        save_csv(secondary_output, secondary_rows, secondary_fields)
                        seek = not playing
                    elif char == "m":
                        matches = [item for item in secondary_rows if item.get("annotation_level") == "MATCH" and item.get("match_id") == row["match_id"]]
                        if matches:
                            value = cycle(matches[0].get("secondary_goal_status", "PENDENTE"))
                            for item in matches:
                                item["secondary_goal_status"] = value
                        save_csv(secondary_output, secondary_rows, secondary_fields)
                        seek = not playing
                    elif char in {"b", "e", "r", "w", "v", "y"}:
                        field = {"b": "ball_visible", "e": "goal_structure_visible", "r": "candidate_goal_association",
                                 "w": "offensive_context", "v": "camera_motion", "y": "occlusion_or_aerial"}[char]
                        cycle_observation(row, field)
                        save_csv(output, rows, fields)
                        seek = not playing
                    elif char == "p":
                        cycle_observation(row, "ball_proposal_quality", ("PENDENTE", "UTIL", "IMPRECISA", "MULTIPLAS", "SEM_PROPOSTA", "INCERTO"))
                        save_csv(output, rows, fields)
                        seek = not playing
                    elif char in CAUSES:
                        tags = set(filter(None, row.get("cause_tags", "").split(";")))
                        tag = CAUSES[char]
                        tags.remove(tag) if tag in tags else tags.add(tag)
                        row["cause_tags"] = ";".join(sorted(tags))
                        save_csv(output, rows, fields)
                        seek = not playing
                    elif char in {"[", ";", "]"}:
                        row["diagnosis_confidence"] = {"[": "BAIXA", ";": "MEDIA", "]": "ALTA"}[char]
                        save_csv(output, rows, fields)
                        seek = not playing
                    elif char == "i":
                        playing = False
                        cv2.destroyWindow(win)
                        row["direct_observations"] = input("Observação direta (o que aparece no vídeo): ").strip()
                        save_csv(output, rows, fields)
                        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                        seek = True
                    elif char == "h":
                        playing = False
                        cv2.destroyWindow(win)
                        row["causal_hypothesis"] = input("Hipótese causal (se houver; não é observação direta): ").strip()
                        save_csv(output, rows, fields)
                        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                        seek = True
                    elif char == "c":
                        if row.get("manual_goal_label") not in {"GOAL", "NO_GOAL", "NOT_EVALUABLE"}:
                            print("Classifique G, N ou U antes de concluir este caso.")
                        else:
                            row["manual_status"] = "COMPLETED"
                            save_csv(output, rows, fields)
                            break
                    elif char in {"a", "d"} or key in {2424832, 2555904}:
                        delta = -round(fps) if char == "a" or key == 2424832 else round(fps)
                        current, playing, seek = max(start_frame, min(end_frame, current + delta)), False, True
                    elif char in {"j", "l"}:
                        delta = -5 if char == "j" else 5
                        current, playing, seek = max(start_frame, min(end_frame, current + delta)), False, True
            finally:
                capture.release()
            cursor += 1
    finally:
        cv2.destroyAllWindows()
    save_csv(output, rows, fields)
    save_csv(secondary_output, secondary_rows, secondary_fields)
    completed = sum(row.get("manual_status") == "COMPLETED" for row in rows)
    print(f"Revisão concluída: {completed}/{len(rows)} casos | {output}")
    return 0


def main() -> int:
    experiment = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=experiment / "manual_case_audit.csv")
    parser.add_argument("--output", type=Path, default=experiment / "manual_case_audit_annotated.csv")
    parser.add_argument("--secondary-goal-manifest", type=Path, default=experiment / "secondary_goal_annotations.csv")
    parser.add_argument("--secondary-goal-output", type=Path, default=experiment / "secondary_goal_annotations_annotated.csv")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--max-preview-width", type=int, default=1440)
    args = parser.parse_args()
    source, fields = read_csv(args.manifest)
    rows, fields = merge_review(source, fields, args.output)
    secondary, secondary_fields = read_csv(args.secondary_goal_manifest)
    frame_sources = {row["match_id"]: float(row.get("source_fps") or 30) for row in source}
    print(f"Casos totais: {len(rows)}; pendentes: {sum(r.get('manual_status') != 'COMPLETED' for r in rows)}")
    print("Atalhos: G/N/U classifica e avanca; F reinicia o clipe completo; Q salva e sai.")
    return annotate(rows, fields, args.videos_dir, args.output, secondary, secondary_fields,
                    args.secondary_goal_output, args.max_preview_width, frame_sources)


if __name__ == "__main__":
    raise SystemExit(main())
