"""Anota manualmente os 12 gols TRAIN perdidos pela shortlist RRF+NMS."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


FIELDS = {
    "1": ("manual_ball_visible", "BOLA VISIVEL"),
    "2": ("manual_clear_offensive_action", "ACAO OFENSIVA CLARA"),
    "3": ("manual_goal_visible", "GOL VISIVEL"),
    "4": ("manual_tracking_problem", "FALHA DETECCAO/CONTINUIDADE"),
}
VALUES = {"y": "SIM", "n": "NAO", "u": "INCERTO"}
REVIEW_STATUSES = {"suppressed_by_nms", "outside_top60_after_nms"}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows, fields = list(reader), reader.fieldnames or []
    required = {"goal_catalog_id", "match_id", "video_filename", "goal_timestamp_seconds", "audit_status", "pool_recoverable"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Manifesto sem colunas obrigatorias: {sorted(missing)}")
    return rows, fields


def merge_existing(source_rows: list[dict[str, str]], fields: list[str], output: Path) -> tuple[list[dict[str, str]], list[str]]:
    extra_fields = ["audit_review_status"]
    for field in extra_fields:
        if field not in fields:
            fields.append(field)
    previous = {}
    if output.is_file():
        with output.open(newline="", encoding="utf-8-sig") as file:
            previous = {row["goal_catalog_id"]: row for row in csv.DictReader(file)}
    rows = []
    for source in source_rows:
        if source.get("pool_recoverable", "").lower() != "true" or source.get("audit_status") not in REVIEW_STATUSES:
            continue
        old = previous.get(source["goal_catalog_id"], {})
        row = dict(source)
        for field in FIELDS:
            row[field] = old.get(field, source.get(field, ""))
        row["manual_notes"] = old.get("manual_notes", source.get("manual_notes", ""))
        row["audit_review_status"] = old.get("audit_review_status", "pending")
        rows.append(row)
    return rows, fields


def save(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def show(image, row: dict[str, str], position: int, total: int, current_seconds: float, goal_offset: float, active_field: str, playing: bool) -> None:
    height, width = image.shape[:2]
    cv2.rectangle(image, (8, 8), (min(width - 8, 1180), min(height - 8, 174)), (0, 0, 0), -1)
    state = "RODANDO" if playing else "PAUSADO"
    cv2.putText(image, f"AUDITORIA TRAIN {position}/{total} | {row['goal_catalog_id']} | {state}", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"Tempo no clipe {current_seconds:.1f}s | gol catalogado aos {goal_offset:.1f}s no clipe | selecione campo 1-4; Y=SIM N=NAO U=INCERTO", (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 230, 255), 1, cv2.LINE_AA)
    labels = []
    for key, (field, title) in FIELDS.items():
        marker = ">" if key == active_field else " "
        labels.append(f"{marker}{key} {title}: {row.get(field) or '-'}")
    cv2.putText(image, " | ".join(labels), (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 240, 180), 1, cv2.LINE_AA)
    cv2.putText(image, "Espaco play/pause | A/D -/+1s | J/L -/+5 frames | C concluir com 4 campos | Q salvar e sair", (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Campo 4: avalie falhas visiveis nas caixas/continuidade; este pipeline nao tem IDs de tracker.", (16, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 230, 255), 1, cv2.LINE_AA)


def annotate(rows: list[dict[str, str]], fields: list[str], clips_dir: Path, output: Path, before: float, max_width: int) -> int:
    pending = [row for row in rows if row.get("audit_review_status") != "completed"]
    if not pending:
        print(f"Auditoria ja concluida: {output}")
        return 0
    window = "Auditoria manual dos gols perdidos TRAIN"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for index, row in enumerate(pending, start=1):
            video_path = clips_dir / f"{row['goal_catalog_id']}.mp4"
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Nao foi possivel abrir o video: {video_path}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
                frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
                duration = frame_count / fps
                goal_offset = min(before, float(row["goal_timestamp_seconds"]))
                start = 0.0
                end = duration
                frame = 0
                end_frame = frame_count - 1
                active = "1"
                playing = False
                seek_required = True
                while True:
                    frame = max(int(start * fps), min(frame, end_frame))
                    if seek_required:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = capture.read()
                    if not ok or image is None:
                        playing = False
                        seek_required = True
                        continue
                    seek_required = False
                    seconds = frame / fps
                    show(image, row, index, len(pending), seconds, goal_offset, active, playing)
                    if image.shape[1] > max_width:
                        scale = max_width / image.shape[1]
                        image = cv2.resize(image, None, fx=scale, fy=scale)
                    cv2.imshow(window, image)
                    key = cv2.waitKeyEx(max(1, round(1000 / fps)) if playing else 0)
                    if key in {-1, 255}:
                        if playing:
                            frame += 1
                            if frame >= end_frame:
                                frame, playing = end_frame, False
                            else:
                                seek_required = False
                        continue
                    char = chr(key & 0xFF).lower()
                    if key == 32 or char == " ":
                        playing = not playing
                        if not playing:
                            seek_required = True
                    elif char == "q" or key == 27:
                        save(output, rows, fields)
                        print(f"Progresso salvo. Retome com o mesmo comando: {output}")
                        return 0
                    elif char in FIELDS:
                        active = char
                        seek_required = not playing
                    elif char in VALUES:
                        field, _ = FIELDS[active]
                        row[field] = VALUES[char]
                        save(output, rows, fields)
                        seek_required = not playing
                    elif char == "c":
                        if all(row.get(field) in VALUES.values() for field, _ in FIELDS.values()):
                            row["audit_review_status"] = "completed"
                            save(output, rows, fields)
                            break
                        print(f"Preencha os quatro campos antes de concluir: {row['goal_catalog_id']}")
                    elif char == "a" or key == 2424832:
                        frame, playing = max(int(start * fps), frame - round(fps)), False
                        seek_required = True
                    elif char == "d" or key == 2555904:
                        frame, playing = min(end_frame, frame + round(fps)), False
                        seek_required = True
                    elif char == "j":
                        frame, playing = max(int(start * fps), frame - 5), False
                        seek_required = True
                    elif char == "l":
                        frame, playing = min(end_frame, frame + 5), False
                        seek_required = True
            finally:
                capture.release()
        save(output, rows, fields)
    finally:
        cv2.destroyAllWindows()
    completed = sum(row.get("audit_review_status") == "completed" for row in rows)
    print(f"Auditoria finalizada: {completed}/{len(rows)} casos | {output}")
    return 0


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    default_dir = base / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_dir / "train_missed_goal_audit_v1.csv")
    parser.add_argument("--clips-dir", type=Path, default=base / "experiments/ball/GOAL-EVENT-MISSED-GOAL-AUDIT-TRAIN-V1/overlay_clips_v1")
    parser.add_argument("--output", type=Path, default=default_dir / "manual_review_v1.csv")
    parser.add_argument("--before", type=float, default=25.0, help="Segundos exibidos antes do timestamp do gol (igual ao gerador dos clipes).")
    parser.add_argument("--max-preview-width", type=int, default=1440)
    args = parser.parse_args()
    if args.before < 0:
        parser.error("--before deve ser >= 0")
    source_rows, fields = read_rows(args.manifest)
    rows, fields = merge_existing(source_rows, fields, args.output)
    print(f"Casos recuperaveis para revisao: {len(rows)} (fora do pool nao entram nesta auditoria)")
    return annotate(rows, fields, args.clips_dir, args.output, args.before, args.max_preview_width)


if __name__ == "__main__":
    raise SystemExit(main())
