"""Anota interativamente quatro pontos da estrutura da goleira nos frames do piloto.

Ordem dos cliques: LT, RT, LB, RB. As coordenadas são armazenadas na resolução
original da imagem, mesmo quando a visualização é redimensionada para a tela.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from experiments.goal_geometry.run_geometry_versions import frame_path

POINT_NAMES = ["lt", "rt", "lb", "rb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anota quatro pontos da goleira.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Diretório das imagens quando o CSV usa match_id/timestamp em vez de image_path.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Anota somente linhas que ainda não estão com status completed.",
    )
    parser.add_argument("--max-preview-width", type=int, default=1280)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    all_rows = list(csv.DictReader(args.input.open(newline="", encoding="utf-8")))
    rows = (
        [row for row in all_rows if row.get("annotation_status") != "completed"]
        if args.pending_only
        else all_rows
    )
    index = 0
    points: list[tuple[int, int]] = []
    scale = 1.0

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(POINT_NAMES):
            points.append((round(x / scale), round(y / scale)))

    window = "Pontos da goleira: LT RT LB RB | S=salvar N=próximo R=limpar Q=sair"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)
    while index < len(rows):
        row = rows[index]
        if row.get("image_path"):
            image_path = Path(row["image_path"])
        else:
            if args.frames_dir is None:
                raise ValueError("Use --frames-dir quando o CSV não possuir image_path")
            timestamp_row = dict(row)
            timestamp_row["timestamp_seconds"] = row.get("timestamp_seconds") or row["timestamp"]
            image_path = frame_path(args.frames_dir, timestamp_row)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Não foi possível abrir: {image_path}")
        scale = min(1.0, args.max_preview_width / image.shape[1])
        points = []
        while True:
            preview = image if scale == 1.0 else cv2.resize(image, None, fx=scale, fy=scale)
            for point_index, (x, y) in enumerate(points):
                px, py = round(x * scale), round(y * scale)
                cv2.circle(preview, (px, py), 6, (0, 0, 255), -1)
                cv2.putText(
                    preview,
                    POINT_NAMES[point_index],
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(window, preview)
            key = cv2.waitKey(50) & 0xFF
            if key == ord("r"):
                points = []
            elif key == ord("s") or key == 13:
                if len(points) != 4:
                    print("Clique exatamente em LT, RT, LB e RB antes de salvar.")
                    continue
                for name, (x, y) in zip(POINT_NAMES, points, strict=True):
                    row[f"{name}_x"] = str(x)
                    row[f"{name}_y"] = str(y)
                if row.get("original_label") == "pending":
                    row["original_label"] = "goal_structure"
                    row["label"] = "goal_structure"
                row["annotation_status"] = "completed"
                row["annotator"] = row.get("annotator", "") or "manual"
                write_rows(args.output, all_rows)
                print(f"Salvo: {row['frame_id']} ({index + 1}/{len(rows)})")
                index += 1
                break
            elif key == ord("n"):
                if row.get("original_label") == "pending":
                    row["original_label"] = "not_goal"
                    row["label"] = "not_goal"
                row["annotation_status"] = "completed"
                row["annotator"] = row.get("annotator", "") or "manual"
                row["notes"] = (row.get("notes", "") + "; marcado como not_goal").lstrip("; ")
                write_rows(args.output, all_rows)
                print(f"Marcado como not_goal: {row['frame_id']} ({index + 1}/{len(rows)})")
                index += 1
                break
            elif key in {ord("q"), 27}:
                write_rows(args.output, all_rows)
                cv2.destroyAllWindows()
                print(f"Progresso salvo em: {args.output}")
                return 0
    cv2.destroyAllWindows()
    print(f"Anotação concluída: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
