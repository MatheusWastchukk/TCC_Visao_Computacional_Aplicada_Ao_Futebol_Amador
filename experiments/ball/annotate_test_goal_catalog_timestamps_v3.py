"""Anota timestamps do catálogo TEST após o freeze V3, com retomada segura."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Registra os timestamps do catálogo TEST após o freeze final V3."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--expected-split", choices=["test"], default="test")
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument(
        "--goal-ids",
        nargs="+",
        help="IDs de gols catalogados a revisar; permite retomar itens já concluídos.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        help="Instante inicial da primeira revisão selecionada, em segundos.",
    )
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()

    freeze_path = Path(__file__).resolve().parents[2] / "experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v3.json"
    if not freeze_path.is_file():
        raise RuntimeError("O freeze V3 precisa existir antes de anotar TEST.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("manifest_version") != 3 or freeze.get("is_final_sprint_freeze") is not True:
        raise RuntimeError("O manifesto V3 não confirma o freeze final.")

    with args.input.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {
        "goal_catalog_id",
        "match_id",
        "video_filename",
        "split",
        "catalog_goal_index",
        "event_timestamp_seconds",
        "annotation_status",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Manifesto sem colunas obrigatórias: {sorted(missing)}")
    if any(row["split"] != args.expected_split for row in rows):
        raise AssertionError(f"Este anotador aceita somente o split {args.expected_split.upper()}.")

    existing: dict[str, dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["goal_catalog_id"]: row for row in csv.DictReader(file)}
    rows = [{field: existing.get(row["goal_catalog_id"], row).get(field, "") for field in fields} for row in rows]
    selected_goal_ids = set(args.goal_ids or [])
    known_goal_ids = {row["goal_catalog_id"] for row in rows}
    unknown_goal_ids = selected_goal_ids - known_goal_ids
    if unknown_goal_ids:
        raise ValueError(f"IDs de gol não encontrados no manifesto: {sorted(unknown_goal_ids)}")
    pending = [
        row
        for row in rows
        if (not selected_goal_ids or row["goal_catalog_id"] in selected_goal_ids)
        and (bool(selected_goal_ids) or not args.pending_only or row["annotation_status"] != "completed")
    ]
    if not pending:
        print("Não há gols pendentes.")
        return 0

    window = "Timestamp de gol catalogado"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    active_match = ""
    carry_frame = 0
    for index, row in enumerate(pending, start=1):
        video_path = args.videos_dir / row["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if row["match_id"] != active_match:
                active_match, carry_frame = row["match_id"], 0
            initial_frame = carry_frame
            if index == 1 and args.start_seconds is not None:
                initial_frame = round(max(0.0, args.start_seconds) * fps)
            current_frame = min(initial_frame, max(0, frame_count - 1))
            playing = False
            while True:
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ok, image = capture.read()
                if not ok or image is None:
                    current_frame = max(0, min(current_frame, frame_count - 1))
                    continue
                seconds = current_frame / fps
                header = (
                    f"{row['match_id']} | gol catalogado {row['catalog_goal_index']} | "
                    f"item {index}/{len(pending)} | {format_time(seconds)}"
                )
                help_text = "ESPACO:play A/D:-/+5s J/L:-/+30s G:registrar U:indefinido X:divergência W:salvar Q:sair"
                cv2.putText(image, header, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(image, help_text, (16, image.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if image.shape[1] > args.max_preview_width:
                    scale = args.max_preview_width / image.shape[1]
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                cv2.imshow(window, image)
                key = cv2.waitKeyEx(max(1, round(1000 / fps)) if playing else 0)
                if key in {-1, 255}:
                    if playing:
                        current_frame = min(frame_count - 1, current_frame + 1)
                    continue
                char = chr(key & 0xFF).lower()
                if char == "q" or key == 27:
                    write_rows(args.output, fields, rows)
                    return 0
                if key == 32 or char == " ":
                    playing = not playing
                elif char == "a" or key == 2424832:
                    current_frame = max(0, current_frame - round(5 * fps))
                    playing = False
                elif char == "d" or key == 2555904:
                    current_frame = min(frame_count - 1, current_frame + round(5 * fps))
                    playing = False
                elif char == "j":
                    current_frame = max(0, current_frame - round(30 * fps))
                    playing = False
                elif char == "l":
                    current_frame = min(frame_count - 1, current_frame + round(30 * fps))
                    playing = False
                elif char == "g":
                    row["event_timestamp_seconds"] = f"{seconds:.3f}"
                    row["event_timestamp_hhmmss"] = format_time(seconds)
                    row["annotation_status"] = "completed"
                    carry_frame = min(frame_count - 1, current_frame + round(3 * fps))
                    break
                elif char == "u":
                    row["event_timestamp_seconds"] = ""
                    row["event_timestamp_hhmmss"] = ""
                    row["annotation_status"] = "not_evaluable"
                    carry_frame = current_frame
                    break
                elif char == "x":
                    row["event_timestamp_seconds"] = ""
                    row["event_timestamp_hhmmss"] = ""
                    row["annotation_status"] = "catalog_discrepancy"
                    carry_frame = current_frame
                    break
                elif char == "w":
                    write_rows(args.output, fields, rows)
                elif playing:
                    current_frame = min(frame_count - 1, current_frame + 1)
        finally:
            capture.release()
    write_rows(args.output, fields, rows)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
