"""Anota contexto visual dos 20 controles NO_GOAL, sem overlays de detectores."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
TAGS = {
    "1": "SEM_ACAO_OFENSIVA",
    "2": "ACAO_OFENSIVA_NAO_GOL",
    "3": "OUTRO_ACAO_FORA_OU_PASSAGEM",
    "4": "SECONDARY_GOAL_VISIVEL",
    "5": "NAO_AVALIAVEL",
    "6": "CONTEXTO_AMBIGUO",
}
CONFIDENCE_KEYS = {"[": "BAIXA", ";": "MEDIA", "]": "ALTA"}
ADDED_FIELDS = (
    "context_tags",
    "direct_observations",
    "taxonomy_confidence",
    "taxonomy_status",
    "mechanism_assessment",
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge(source_path: Path, output_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    source, fields = read_csv(source_path)
    old_rows: dict[str, dict[str, str]] = {}
    if output_path.is_file():
        previous, _ = read_csv(output_path)
        old_rows = {row["control_group_uid"]: row for row in previous}
    for field in ADDED_FIELDS:
        if field not in fields:
            fields.append(field)
    merged: list[dict[str, str]] = []
    for item in source:
        row = dict(item)
        old = old_rows.get(item["control_group_uid"], {})
        for field in ADDED_FIELDS:
            if field in old:
                row[field] = old[field]
            elif field == "mechanism_assessment":
                row[field] = "PENDENTE_REQUER_EVIDENCIA_SEPARADA"
            else:
                row[field] = ""
        merged.append(row)
    return merged, fields


def toggle_tag(row: dict[str, str], tag: str) -> None:
    values = set(filter(None, row.get("context_tags", "").split(";")))
    if tag in values:
        values.remove(tag)
    else:
        values.add(tag)
    row["context_tags"] = ";".join(sorted(values))


def draw(image, row: dict[str, str], index: int, total: int, seconds: float,
         playing: bool, speed: float, width_limit: int):
    height, width = image.shape[:2]
    panel_h = min(250, max(205, height // 4))
    cv2.rectangle(image, (0, 0), (width, panel_h), (0, 0, 0), -1)
    tags = row.get("context_tags") or "(sem categoria)"
    label = row.get("manual_label") or "SEM_ROTULO"
    lines = [
        f"CONTROLE {index}/{total} | {row['match_id']} | rank RF #{row['control_rank_full_pool']} | {'PLAY' if playing else 'PAUSE'} {speed:g}x",
        f"Tempo {seconds:.1f}s | candidato {float(row['control_candidate_timestamp_seconds']):.1f}s | decisao previa {label}",
        "Video original, sem overlays. Categorize apenas o contexto visivel; nao infira causa do detector.",
        "1 sem acao | 2 acao sem gol | 3 outra (fora/passagem) | 4 secondary goal visivel",
        "5 nao avaliavel | 6 contexto ambiguo | categorias podem ser combinadas",
        "SPACE play/pause | = velocidade 1/2/4x | A/D -/+5s | I observacao direta | [ ; ] confianca",
        "C conclui taxonomia e avanca | Q salva/sai | categorias atuais: " + tags,
    ]
    for line_index, line in enumerate(lines):
        cv2.putText(image, line[:200], (10, 23 + line_index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    if image.shape[1] > width_limit:
        image = cv2.resize(image, None, fx=width_limit / image.shape[1], fy=width_limit / image.shape[1])
    return image


def annotate(rows: list[dict[str, str]], fields: list[str], videos_dir: Path,
             output: Path, width_limit: int) -> int:
    pending = [i for i, row in enumerate(rows) if row.get("taxonomy_status") != "COMPLETED"]
    if not pending:
        print(f"Taxonomia completa: {output}")
        return 0
    title = "Taxonomia de contexto — controles NO_GOAL"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        cursor = 0
        while cursor < len(pending):
            row = rows[pending[cursor]]
            video_path = videos_dir / row["video_filename"]
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Video TRAIN nao encontrado: {video_path}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
                frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
                start = max(0, min(frame_count - 1, int(float(row["delivery_start_seconds"]) * fps)))
                end = max(start, min(frame_count - 1, int(float(row["delivery_end_seconds"]) * fps)))
                frame = start
                playing, seek, speed = False, True, 2.0
                while True:
                    frame = max(start, min(frame, end))
                    if seek:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = capture.read()
                    if not ok or image is None:
                        playing, seek = False, True
                        continue
                    seek = False
                    seconds = frame / fps
                    shown = draw(image, row, cursor + 1, len(pending), seconds,
                                 playing, speed, width_limit)
                    cv2.imshow(title, shown)
                    key = cv2.waitKeyEx(max(1, round(1000 / (fps * speed))) if playing else 0)
                    if key in {-1, 255}:
                        if playing:
                            frame += max(1, round(speed))
                            if frame >= end:
                                frame, playing, seek = end, False, True
                        continue
                    char = chr(key & 0xFF).lower()
                    if key == 32 or char == " ":
                        playing = not playing
                        seek = not playing
                    elif char == "q" or key == 27:
                        write_csv(output, rows, fields)
                        print(f"Progresso salvo; retome pelo mesmo comando: {output}")
                        return 0
                    elif char == "=":
                        speed = {1.0: 2.0, 2.0: 4.0, 4.0: 1.0}[speed]
                    elif char in TAGS:
                        toggle_tag(row, TAGS[char])
                        write_csv(output, rows, fields)
                    elif char in CONFIDENCE_KEYS:
                        row["taxonomy_confidence"] = CONFIDENCE_KEYS[char]
                        write_csv(output, rows, fields)
                    elif char == "i":
                        playing = False
                        cv2.destroyWindow(title)
                        note = input(f"Observacao direta para {row['control_group_uid']}: ").strip()
                        if note:
                            row["direct_observations"] = note
                        write_csv(output, rows, fields)
                        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                        seek = True
                    elif char == "c":
                        if not row.get("context_tags"):
                            print("Marque ao menos uma categoria (use 6 se o contexto permanecer ambiguo).")
                            continue
                        row["taxonomy_status"] = "COMPLETED"
                        write_csv(output, rows, fields)
                        break
                    elif char in {"a", "d"} or key in {2424832, 2555904}:
                        delta = -round(5 * fps) if char == "a" or key == 2424832 else round(5 * fps)
                        frame = max(start, min(end, frame + delta))
                        playing, seek = False, True
            finally:
                capture.release()
            cursor += 1
    finally:
        cv2.destroyAllWindows()
    write_csv(output, rows, fields)
    done = sum(row.get("taxonomy_status") == "COMPLETED" for row in rows)
    print(f"Taxonomia concluida: {done}/{len(rows)} | {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=EXPERIMENT / "pairwise_control_audit_v1_annotated.csv")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "control_context_taxonomy_v1.csv")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--max-preview-width", type=int, default=1440)
    parser.add_argument("--prepare-only", action="store_true", help="Cria/atualiza CSV retomavel sem abrir videos.")
    args = parser.parse_args()
    rows, fields = merge(args.manifest, args.output)
    write_csv(args.output, rows, fields)
    completed = sum(row.get("taxonomy_status") == "COMPLETED" for row in rows)
    print(f"Controles: {len(rows)}; taxonomias concluidas: {completed}; pendentes: {len(rows)-completed}")
    if args.prepare_only:
        return 0
    return annotate(rows, fields, args.videos_dir, args.output, args.max_preview_width)


if __name__ == "__main__":
    raise SystemExit(main())
