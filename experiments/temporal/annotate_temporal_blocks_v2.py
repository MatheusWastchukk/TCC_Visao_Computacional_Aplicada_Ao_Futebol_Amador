"""Anota intervalos temporais de blocos contínuos usando uma janela de vídeo."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2


LABELS = {
    ord("1"): ("goal_structure", "main"),
    ord("2"): ("not_goal", "unknown"),
    ord("3"): ("inconclusive", "unknown"),
    ord("4"): ("not_evaluable", "unknown"),
}
LABEL_NAMES = {
    "goal_structure": "estrutura de gol",
    "not_goal": "sem estrutura",
    "inconclusive": "inconclusivo",
    "not_evaluable": "não avaliável",
}
ARROW_LEFT = {81, 2424832}
ARROW_RIGHT = {83, 2555904}
ARROW_UP = {82, 2490368}
ARROW_DOWN = {84, 2621440}
ANNOTATION_FIELDS = [
    "interval_id",
    "block_id",
    "match_id",
    "video_filename",
    "start_seconds",
    "end_seconds",
    "start_time_hhmmss",
    "end_time_hhmmss",
    "interval_label",
    "structure_scope",
    "annotation_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anota intervalos de blocos contínuos de vídeo temporal V2."
    )
    parser.add_argument("--blocks", type=Path, required=True, help="CSV dos blocos contínuos.")
    parser.add_argument("--output", type=Path, required=True, help="CSV de intervalos anotados.")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/Videos_Edited"),
        help="Diretório que contém os vídeos.",
    )
    parser.add_argument("--max-preview-width", type=int, default=1280)
    parser.add_argument("--preview-fps", type=float, default=10.0)
    return parser.parse_args()


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        # Ignora colunas extras de CSVs antigos que contenham vírgulas sem aspas.
        normalized_rows = [{field: row.get(field, "") or "" for field in fields} for row in rows]
        writer.writerows(normalized_rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def draw_text(frame, text: str, origin: tuple[int, int], color=(255, 255, 255)) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)


def main() -> int:
    args = parse_args()
    blocks = read_rows(args.blocks)
    if not blocks:
        raise ValueError(f"Nenhum bloco encontrado em: {args.blocks}")

    annotations = read_rows(args.output)
    annotations = [row for row in annotations if row.get("annotation_status") == "completed"]
    completed_by_block = {row["block_id"] for row in annotations}
    completed_by_block.update(
        row["block_id"]
        for row in blocks
        if row.get("annotation_status") == "completed"
    )
    block_index = next(
        (index for index, row in enumerate(blocks) if row["block_id"] not in completed_by_block),
        0,
    )
    current_label = "goal_structure"
    current_scope = "main"
    interval_start: float | None = None
    playing = False
    cap: cv2.VideoCapture | None = None
    current_video: Path | None = None
    fps = 30.0
    current_second = 0.0
    last_frame = None
    auto_mode = True
    space_key_down = False

    window = "Anotador temporal V2"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def save_progress() -> None:
        write_csv(args.output, ANNOTATION_FIELDS, annotations)
        block_fields = [field for field in blocks[0] if field is not None]
        write_csv(args.blocks, block_fields, blocks)

    def close_video() -> None:
        nonlocal cap, current_video
        if cap is not None:
            cap.release()
        cap = None
        current_video = None

    def open_block(index: int) -> None:
        nonlocal cap, current_video, fps, current_second, last_frame, interval_start
        close_video()
        row = blocks[index]
        video_path = args.videos_dir / row["video_filename"]
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        current_video = video_path
        block_start = float(row["start_seconds"])
        saved_ends = [
            float(annotation["end_seconds"])
            for annotation in annotations
            if annotation.get("block_id") == row["block_id"]
        ]
        current_second = max([block_start, *saved_ends])
        interval_start = None
        last_frame = None

    def read_at(second: float):
        nonlocal last_frame
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return last_frame
        last_frame = frame
        return frame

    def append_interval(end_second: float | None = None, keep_open: bool = False) -> None:
        nonlocal interval_start
        if interval_start is None:
            print("Marque o início com [ antes de fechar o intervalo com ].")
            return
        row = blocks[block_index]
        final_second = current_second if end_second is None else end_second
        if final_second <= interval_start:
            print("O fim precisa ser posterior ao início.")
            return
        interval_id = f"temporal_v2_interval_{len(annotations) + 1:03d}"
        annotations.append(
            {
                "interval_id": interval_id,
                "block_id": row["block_id"],
                "match_id": row["match_id"],
                "video_filename": row["video_filename"],
                "start_seconds": f"{interval_start:.3f}",
                "end_seconds": f"{final_second:.3f}",
                "start_time_hhmmss": format_time(interval_start),
                "end_time_hhmmss": format_time(final_second),
                "interval_label": current_label,
                "structure_scope": current_scope,
                "annotation_status": "completed",
                "notes": "",
            }
        )
        completed_by_block.add(row["block_id"])
        row["annotation_status"] = "completed"
        interval_start = final_second if keep_open else None
        save_progress()
        print(f"Intervalo salvo: {interval_id} ({format_time(float(annotations[-1]['start_seconds']))}–{format_time(final_second)})")

    try:
        open_block(block_index)
        while True:
            block = blocks[block_index]
            start = float(block["start_seconds"])
            end = float(block["end_seconds"])
            current_second = min(max(current_second, start), end)
            if playing and cap is not None:
                frames_to_advance = max(1, round(fps / args.preview_fps))
                for _ in range(frames_to_advance - 1):
                    cap.grab()
                ok, frame = cap.read()
                if ok:
                    last_frame = frame
                    current_second += frames_to_advance / fps
                else:
                    frame = last_frame
            else:
                frame = read_at(current_second)
            if frame is None:
                raise RuntimeError(f"Não foi possível ler o frame em {current_second:.3f}s.")
            scale = min(1.0, args.max_preview_width / frame.shape[1])
            preview = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale)
            status = "REPRODUZINDO" if playing else "PAUSADO"
            label_text = LABEL_NAMES[current_label]
            mode_text = "automático" if auto_mode else "manual"
            draw_text(preview, f"Bloco {block_index + 1}/{len(blocks)} | {block['block_id']}", (15, 30), (0, 255, 255))
            draw_text(preview, f"Tempo: {format_time(current_second)} / {format_time(end)} | {status}", (15, 60))
            draw_text(preview, f"Rótulo: {label_text} | Escopo: {current_scope}", (15, 90), (0, 255, 0))
            draw_text(preview, "1 estrutura  2 sem estrutura  3 inconclusivo  4 não avaliável", (15, 120), (255, 220, 120))
            draw_text(preview, "M principal  R secundária  T múltiplas | [ início  ] fim/salvar", (15, 150), (255, 220, 120))
            draw_text(preview, "espaço play  ←→ 1s  ↑↓ 5s  N/P bloco  S salvar  Q sair", (15, 180), (255, 220, 120))
            if interval_start is not None:
                draw_text(preview, f"Intervalo aberto desde {format_time(interval_start)}", (15, 210), (0, 165, 255))
            cv2.imshow(window, preview)
            delay = max(1, round(1000.0 / args.preview_fps)) if playing else 80
            key = cv2.waitKeyEx(delay)
            if playing and current_second >= end:
                current_second = end
                playing = False
            if key == ord(" "):
                if not space_key_down:
                    playing = not playing
                space_key_down = True
                continue
            space_key_down = False
            if key in {-1, 255}:
                continue
            if key in {ord("q"), 27}:
                save_progress()
                break
            if key in LABELS:
                new_label, new_scope = LABELS[key]
                if auto_mode:
                    if interval_start is None:
                        interval_start = current_second
                    elif new_label != current_label or new_scope != current_scope:
                        append_interval(current_second, keep_open=True)
                current_label, current_scope = new_label, new_scope
            elif key == ord("m"):
                if auto_mode and interval_start is not None and current_scope != "main":
                    append_interval(current_second, keep_open=True)
                current_scope = "main"
            elif key == ord("r"):
                if auto_mode and interval_start is not None and current_scope != "secondary":
                    append_interval(current_second, keep_open=True)
                current_scope = "secondary"
            elif key == ord("t"):
                if auto_mode and interval_start is not None and current_scope != "multiple":
                    append_interval(current_second, keep_open=True)
                current_scope = "multiple"
            elif key == ord("a"):
                auto_mode = not auto_mode
            elif key == ord("["):
                interval_start = current_second
            elif key == ord("]"):
                append_interval()
            elif key == ord("e"):
                current_second = end
                append_interval()
            elif key == ord("s"):
                save_progress()
                print(f"Progresso salvo em: {args.output}")
            elif key in ARROW_LEFT:
                playing = False
                current_second = max(start, current_second - 1.0)
            elif key in ARROW_RIGHT:
                playing = False
                current_second = min(end, current_second + 1.0)
            elif key in ARROW_UP:
                playing = False
                current_second = min(end, current_second + 5.0)
            elif key in ARROW_DOWN:
                playing = False
                current_second = max(start, current_second - 5.0)
            elif key == ord(","):
                playing = False
                current_second = max(start, current_second - 1.0 / fps)
            elif key == ord("."):
                playing = False
                current_second = min(end, current_second + 1.0 / fps)
            elif key in {ord("n"), ord("p")}:
                save_progress()
                direction = 1 if key == ord("n") else -1
                next_index = block_index + direction
                if 0 <= next_index < len(blocks):
                    block_index = next_index
                    open_block(block_index)
                else:
                    print("Não há outro bloco nessa direção.")
    finally:
        close_video()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
