"""Anota candidatos comparadores da Fase D usando vídeo original sem overlays."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"g": "GOAL", "n": "NO_GOAL", "u": "NOT_EVALUABLE"}


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


def merge(source: list[dict[str, str]], fields: list[str], output: Path) -> tuple[list[dict[str, str]], list[str]]:
    for field, default in (("manual_label", ""), ("manual_status", "PENDING"), ("reviewer_note", "")):
        if field not in fields:
            fields.append(field)
    previous = {}
    if output.is_file():
        old, _ = read_csv(output)
        previous = {row["control_group_uid"]: row for row in old}
    rows = []
    for item in source:
        row = dict(item)
        old = previous.get(item["control_group_uid"], {})
        for field, default in (("manual_label", ""), ("manual_status", "PENDING"), ("reviewer_note", "")):
            row[field] = old.get(field, item.get(field, default))
        rows.append(row)
    return rows, fields


def draw(
    image,
    row: dict[str, str],
    index: int,
    total: int,
    seconds: float,
    state: str,
    speed: float,
    width_limit: int,
    height_limit: int,
):
    height, width = image.shape[:2]
    panel_h = min(190, max(155, height // 5))
    cv2.rectangle(image, (0, 0), (width, panel_h), (0, 0, 0), -1)
    lines = [
        f"COMPARADOR {index}/{total} | {row['match_id']} | rank RF #{row['control_rank_full_pool']} | {state}",
        f"Posição no vídeo: {seconds:.1f}s | candidato {float(row['control_candidate_timestamp_seconds']):.1f}s | velocidade {speed:g}x",
        (f"Comparado ao miss: {row['goal_catalog_id']} (rank #{row['missed_goal_rank_full_pool']})"
         if row.get("review_context") != "TOP60_UNCOVERED_WINDOW_NEAREST_CATALOG_GOAL"
         else f"Janela top-60 sem timestamp de gol catalogado; par de referÃªncia: {row['goal_catalog_id']} (rank #{row['missed_goal_rank_full_pool']})"),
        "Vídeo original, sem caixas BALL/GOAL. Classifique apenas o que vê nesta janela.",
        "G=GOAL (gol visível) | N=NO_GOAL (avaliável, sem gol) | U=NOT_EVALUABLE",
        "SPACE play/pause | = velocidade 1x/2x/4x | A/D -/+1s | I=nota | Q salvar e sair",
        f"Decisão atual: {row.get('manual_label') or 'PENDENTE'} | estado: {row.get('manual_status','PENDING')}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(image, line[:190], (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA)
    scale = min(1.0, width_limit / image.shape[1], height_limit / image.shape[0])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def annotate(
    rows: list[dict[str, str]],
    fields: list[str],
    videos_dir: Path,
    output: Path,
    width_limit: int,
    height_limit: int,
    window_title: str,
) -> int:
    pending = [i for i, row in enumerate(rows) if row.get("manual_status") != "COMPLETED"]
    if not pending:
        print(f"Auditoria concluída: {output}")
        return 0
    title = window_title
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    window_size = None
    try:
        for position, row_index in enumerate(pending, start=1):
            row = rows[row_index]
            video_path = videos_dir / row["video_filename"]
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
                frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
                duration = frame_count / fps
                start = max(0.0, float(row["delivery_start_seconds"]))
                end = min(duration, float(row["delivery_end_seconds"]))
                start_frame = min(frame_count - 1, int(start * fps))
                end_frame = min(frame_count - 1, int(end * fps))
                current = start_frame
                playing = False
                seek = True
                speed = 2.0
                while True:
                    current = max(start_frame, min(current, end_frame))
                    if seek:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, current)
                    ok, image = capture.read()
                    if not ok or image is None:
                        playing, seek = False, True
                        continue
                    seek = False
                    seconds = current / fps
                    shown = draw(
                        image,
                        row,
                        position,
                        len(pending),
                        seconds,
                        "RODANDO" if playing else "PAUSADO",
                        speed,
                        width_limit,
                        height_limit,
                    )
                    cv2.imshow(title, shown)
                    shown_size = (shown.shape[1], shown.shape[0])
                    if shown_size != window_size:
                        cv2.resizeWindow(title, *shown_size)
                        cv2.moveWindow(title, 30, 30)
                        window_size = shown_size
                    key = cv2.waitKeyEx(max(1, round(1000 / (fps * speed))) if playing else 0)
                    if key in {-1, 255}:
                        if playing:
                            current += max(1, round(speed))
                            if current >= end_frame:
                                current, playing, seek = end_frame, False, True
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
                    elif char in LABELS:
                        row["manual_label"] = LABELS[char]
                        row["manual_status"] = "COMPLETED"
                        write_csv(output, rows, fields)
                        break
                    elif char == "i":
                        playing = False
                        cv2.destroyWindow(title)
                        row["reviewer_note"] = input(f"Nota curta para {row['control_group_uid']} (Enter para manter): ").strip() or row.get("reviewer_note", "")
                        write_csv(output, rows, fields)
                        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                        seek = True
                    elif char in {"a", "d"} or key in {2424832, 2555904}:
                        delta = -round(fps) if char == "a" or key == 2424832 else round(fps)
                        current = max(start_frame, min(end_frame, current + delta))
                        playing, seek = False, True
            finally:
                capture.release()
    finally:
        cv2.destroyAllWindows()
    write_csv(output, rows, fields)
    completed = sum(row.get("manual_status") == "COMPLETED" for row in rows)
    print(f"Auditoria finalizada: {completed}/{len(rows)} | {output}")
    return 0


def main() -> int:
    experiment = ROOT / "experiments/ball/GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=experiment / "pairwise_control_audit_v1.csv")
    parser.add_argument("--output", type=Path, default=experiment / "pairwise_control_audit_v1_annotated.csv")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "data/Videos_Edited")
    parser.add_argument("--max-preview-width", type=int, default=1440)
    parser.add_argument("--max-preview-height", type=int, default=800)
    parser.add_argument("--window-title", default="Auditoria de comparadores da Fase D — sem overlays")
    args = parser.parse_args()
    source, fields = read_csv(args.manifest)
    rows, fields = merge(source, fields, args.output)
    print(f"Comparadores: {len(rows)}; pendentes: {sum(r.get('manual_status') != 'COMPLETED' for r in rows)}")
    print("G/N/U classifica e avança. Use U se não conseguir decidir; Q salva e sai.")
    return annotate(
        rows,
        fields,
        args.videos_dir,
        args.output,
        args.max_preview_width,
        args.max_preview_height,
        args.window_title,
    )


if __name__ == "__main__":
    raise SystemExit(main())
