"""Anota clipes e evidências do GOAL-EVENT-DEVSET-V1 com retomada segura."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


EVIDENCE_FIELDS = [
    "ball_visible",
    "goal_visible",
    "spatial_candidate",
    "relative_motion_valid",
    "identity_stable",
    "center_visible_after",
    "ball_at_center_after",
    "players_reset",
    "kickoff_observed",
    "scoreboard_visible",
    "score_changed",
]
REASON_KEYS = {
    "1": "BALL_NOT_VISIBLE",
    "2": "GOAL_NOT_VISIBLE",
    "3": "CAMERA_LOST",
    "4": "OCLUSAO_CRITICA",
    "5": "TRAJETORIA_INSUFICIENTE",
    "6": "MULTIPLE_BALLS",
    "7": "OUTRO",
}
TOGGLE_KEYS = {
    "b": "ball_visible",
    "o": "goal_visible",
    "c": "spatial_candidate",
    "m": "relative_motion_valid",
    "i": "identity_stable",
    "z": "center_visible_after",
    "h": "ball_at_center_after",
    "p": "players_reset",
    "k": "kickoff_observed",
    "s": "scoreboard_visible",
    "d": "score_changed",
}


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def output_fields(source_fields: list[str]) -> list[str]:
    fields = list(source_fields)
    if "ground_truth" not in fields:
        fields.append("ground_truth")
    if "source_ground_truth" not in fields:
        fields.append("source_ground_truth")
    for field in EVIDENCE_FIELDS + ["not_evaluable_reason", "annotation_status", "notes"]:
        if field not in fields:
            fields.append(field)
    return fields


def save(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def initial_row(source: dict[str, str], fields: list[str]) -> dict[str, str]:
    row = {field: source.get(field, "") for field in fields}
    row["source_ground_truth"] = source.get("ground_truth", "")
    row["ground_truth"] = ""
    row["annotation_status"] = "pending"
    for field in EVIDENCE_FIELDS:
        row[field] = ""
    row["not_evaluable_reason"] = ""
    return row


def toggle(value: str) -> str:
    return {"": "True", "True": "False", "False": ""}.get(value, "True")


def status(field: str, row: dict[str, str]) -> str:
    return {"True": "S", "False": "N", "": "-"}[row.get(field, "")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anota clipes do GOAL-EVENT-DEVSET-V1 sem sobrescrever o manifesto de entrada."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-seconds", type=float, default=10.0)
    parser.add_argument("--post-seconds", type=float, default=20.0)
    parser.add_argument(
        "--playback-offset-seconds",
        type=float,
        default=0.0,
        help="Deslocamento aplicado somente à janela exibida, sem alterar o instante do evento.",
    )
    parser.add_argument(
        "--event-ids",
        nargs="+",
        default=None,
        help="Revisa somente os event_id informados e preserva os demais registros no CSV de saída.",
    )
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--max-preview-width", type=int, default=1280)
    args = parser.parse_args()
    if args.pre_seconds < 0 or args.post_seconds <= 0:
        raise ValueError("--pre-seconds deve ser não negativo e --post-seconds deve ser positivo.")

    with args.input.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        source_rows = list(reader)
        source_fields = reader.fieldnames or []
    if not source_rows:
        raise ValueError("O manifesto de entrada não possui eventos.")
    required = {"event_id", "video_filename", "event_timestamp_seconds"}
    missing = required - set(source_fields)
    if missing:
        raise ValueError(f"Manifesto sem colunas obrigatórias: {sorted(missing)}")
    disallowed = sorted({row.get("split", "SEM_SPLIT") for row in source_rows if row.get("split") != "train"})
    if disallowed:
        raise AssertionError(
            "GOAL-EVENT-DEVSET-V1 aceita somente TRAIN; splits encontrados: "
            f"{', '.join(disallowed)}. TEST é reservado para a avaliação final."
        )

    fields = output_fields(source_fields)
    existing: dict[str, dict[str, str]] = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = {row["event_id"]: row for row in csv.DictReader(file)}
    rows = [
        {field: existing[source["event_id"]].get(field, "") for field in fields}
        if source["event_id"] in existing
        else initial_row(source, fields)
        for source in source_rows
    ]
    available_ids = {row["event_id"] for row in rows}
    selected_ids = set(args.event_ids) if args.event_ids else available_ids
    unknown_ids = sorted(selected_ids - available_ids)
    if unknown_ids:
        raise ValueError(f"event_id ausente do manifesto: {unknown_ids}")
    selected_rows = [row for row in rows if row["event_id"] in selected_ids]
    pending = [
        row for row in selected_rows
        if not args.pending_only or row["annotation_status"] != "completed" or not row.get("ground_truth")
    ]
    if not pending:
        print("Não há eventos pendentes.")
        return 0

    window = "Anotação de clipes: eventos de gol"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for event_index, row in enumerate(pending, start=1):
        video_path = args.videos_dir / row["video_filename"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            event_time = float(row["event_timestamp_seconds"])
            playback_time = event_time + args.playback_offset_seconds
            first_frame = max(0, round((playback_time - args.pre_seconds) * fps))
            last_frame = max(first_frame, round((playback_time + args.post_seconds) * fps))
            current_frame, playing = first_frame, False
            while True:
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Não foi possível ler {video_path} no quadro {current_frame}.")
                seconds = current_frame / fps
                decision = row.get("ground_truth") or "SEM RÓTULO"
                evidence = " ".join(
                    f"{key.upper()}:{status(field, row)}" for key, field in TOGGLE_KEYS.items()
                )
                reason = row.get("not_evaluable_reason") or "-"
                state = "RODANDO" if playing else "PAUSADO"
                cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1250), 122), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    f"{event_index}/{len(pending)} | {row.get('match_id', '-')} | {format_time(seconds)} | {state} | {decision}",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "Espaço tocar | A/D +/-1s | J/L +/-5 quadros | G gol | N não gol | U não avaliável | W salvar | Q sair",
                    (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 220, 120), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "Evidências: B bola O gol C espacial M movimento I identidade Z centro H bola-centro P jogadores K kickoff S placar D mudança",
                    (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 240, 180), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"{evidence} | motivo não avaliável: {reason} | 1-7 define motivo | R limpa evidências",
                    (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 240, 180), 1, cv2.LINE_AA,
                )
                scale = min(1.0, args.max_preview_width / image.shape[1])
                canvas = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
                cv2.imshow(window, canvas)
                key = cv2.waitKeyEx(max(1, round(1000 / fps)) if playing else 0)
                if key in {-1, 255}:
                    if playing:
                        current_frame = min(last_frame, current_frame + 1)
                        if current_frame == last_frame:
                            playing = False
                    continue
                char = chr(key & 0xFF).lower()
                if char == "q" or key == 27:
                    save(args.output, rows, fields)
                    cv2.destroyAllWindows()
                    print(f"Progresso salvo: {args.output}")
                    return 0
                if key == 32:
                    playing = not playing
                elif char == "a" or key == 2424832:
                    current_frame, playing = max(first_frame, current_frame - round(fps)), False
                elif char == "d" or key == 2555904:
                    current_frame, playing = min(last_frame, current_frame + round(fps)), False
                elif char == "j":
                    current_frame, playing = max(first_frame, current_frame - 5), False
                elif char == "l":
                    current_frame, playing = min(last_frame, current_frame + 5), False
                elif char in TOGGLE_KEYS:
                    field = TOGGLE_KEYS[char]
                    row[field] = toggle(row.get(field, ""))
                elif char in REASON_KEYS:
                    row["not_evaluable_reason"] = REASON_KEYS[char]
                elif char == "r":
                    for field in EVIDENCE_FIELDS:
                        row[field] = ""
                    row["not_evaluable_reason"] = ""
                elif char in {"g", "n", "u"}:
                    row["ground_truth"] = {"g": "GOAL", "n": "NO_GOAL", "u": "NOT_EVALUABLE"}[char]
                    if char != "u":
                        row["not_evaluable_reason"] = ""
                    row["annotation_status"] = "completed"
                    save(args.output, rows, fields)
                    print(f"Salvo: {row['event_id']} -> {row['ground_truth']}")
                    break
                elif char == "w":
                    save(args.output, rows, fields)
                    print(f"Progresso salvo: {args.output}")
        finally:
            capture.release()
    cv2.destroyAllWindows()
    print(f"Anotação concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
