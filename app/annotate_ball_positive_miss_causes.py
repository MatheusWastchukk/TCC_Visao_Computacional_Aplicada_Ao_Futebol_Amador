"""Review observable visual factors on frozen positive BALL miss frames."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
TAGS = {
    ord("1"): "SMALL_BALL",
    ord("2"): "BLUR",
    ord("3"): "OCCLUSION_PLAYER",
    ord("4"): "OCCLUSION_GOALKEEPER",
    ord("5"): "OCCLUSION_NET",
    ord("6"): "LOW_CONTRAST",
    ord("7"): "FRAME_EDGE",
    ord("8"): "BAD_LOCALIZATION",
    ord("9"): "OTHER",
    ord("0"): "UNCERTAIN",
}


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
    defaults = {
        "measured_flags": "",
        "visual_cause_tags": "",
        "visual_evidence_note": "",
        "visual_review_status": "PENDING",
    }
    for field in defaults:
        if field not in fields:
            fields.append(field)
    prior = {}
    if output.is_file():
        previous, _ = read_csv(output)
        prior = {row["sequence_id"] + "|" + row["sequence_frame"]: row for row in previous}
    rows = []
    for item in source:
        key = item["sequence_id"] + "|" + item["sequence_frame"]
        old = prior.get(key, {})
        row = dict(item)
        area = float(item.get("gt_box_area_px2") or 0)
        flags = []
        if int(float(item.get("candidate_count") or 0)) == 0:
            flags.append("NO_TOP10_PROPOSAL")
        if area < 400:
            flags.append("GT_BOX_AREA_LT_400PX2")
        row["measured_flags"] = old.get("measured_flags", ";".join(flags))
        for field in ("visual_cause_tags", "visual_evidence_note", "visual_review_status"):
            row[field] = old.get(field, defaults[field])
        rows.append(row)
    return rows, fields


def parse_box(raw: str) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    box = json.loads(raw)
    if not isinstance(box, list) or len(box) != 4:
        return None
    return tuple(float(v) for v in box)


def render(row: dict[str, str], image_path: Path, index: int, total: int, width_limit: int):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    original = image.copy()
    gt = parse_box(row.get("ball_of_play_box_json", ""))
    tracker = parse_box(row.get("selected_box_json", ""))
    if gt:
        x1, y1, x2, y2 = map(round, gt)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        margin = 150
        h, w = original.shape[:2]
        crop = original[max(0, y1-margin):min(h, y2+margin), max(0, x1-margin):min(w, x2+margin)]
    else:
        crop = original
    if tracker:
        x1, y1, x2, y2 = map(round, tracker)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    tag_values = [v for v in (row.get("visual_cause_tags") or "").split(";") if v]
    flags = row.get("measured_flags") or "(none)"
    panel_h = 150
    cv2.rectangle(image, (0, 0), (image.shape[1], panel_h), (0, 0, 0), -1)
    lines = [
        f"MISS {index}/{total} | {row['sequence_id']} frame {row['sequence_frame']} | {row['match_id']} {row['timestamp_hhmmss']}",
        f"GT={row['ball_visibility']} {row.get('gt_box_width_px')}x{row.get('gt_box_height_px')} px; tracker IoU={row.get('selected_tracker_box_iou') or 'NA'}; proposals={row.get('candidate_count')}",
        f"MENSURADO (nao e causa): {flags}",
        "GT amarela | tracker vermelha | tags podem se sobrepor; selecione apenas evidencia visivel no frame.",
        "1 pequena 2 blur 3 jogador 4 goleiro 5 rede 6 contraste 7 borda 8 localizacao 9 outro 0 incerto",
        "Tecla da tag alterna | Enter conclui/avanca | Backspace volta | R limpa tags | Q salva e sai",
        "Tags atuais: " + (", ".join(tag_values) if tag_values else "(nenhuma)"),
    ]
    for i, text in enumerate(lines):
        cv2.putText(image, text[:190], (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
    h, w = image.shape[:2]
    scale = min(1.0, (width_limit * 0.62) / w)
    image = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    if crop.size:
        crop_h, crop_w = crop.shape[:2]
        target_h = max(1, round(crop_h * min(2.0, width_limit * 0.36 / crop_w)))
        crop = cv2.resize(crop, (max(1, round(crop_w * target_h / crop_h)), target_h))
    return image, crop


def annotate(
    rows: list[dict[str, str]],
    fields: list[str],
    output: Path,
    width_limit: int,
    height_limit: int,
    window_title: str,
) -> int:
    pending = [i for i, row in enumerate(rows) if row.get("visual_review_status") != "COMPLETED"]
    if not pending:
        print(f"Auditoria visual concluida: {output}")
        return 0
    title = window_title
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    window_size = None
    cursor = 0
    try:
        while cursor < len(pending):
            row_index = pending[cursor]
            row = rows[row_index]
            frame_path = ROOT / row["image_path"]
            full, crop = render(row, frame_path, cursor + 1, len(pending), width_limit)
            fh, fw = full.shape[:2]
            canvas_w = min(width_limit, fw + crop.shape[1] + 16)
            canvas_h = max(fh, crop.shape[0])
            canvas = cv2.copyMakeBorder(full, 0, canvas_h - fh, 0, max(0, canvas_w - fw), cv2.BORDER_CONSTANT, value=(30, 30, 30))
            if crop.size:
                x = min(fw + 8, canvas_w - crop.shape[1])
                if x >= 0:
                    y = max(0, (canvas_h - crop.shape[0]) // 2)
                    canvas[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
            scale = min(1.0, width_limit / canvas.shape[1], height_limit / canvas.shape[0])
            if scale < 1.0:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(title, canvas)
            shown_size = (canvas.shape[1], canvas.shape[0])
            if shown_size != window_size:
                cv2.resizeWindow(title, *shown_size)
                cv2.moveWindow(title, 30, 30)
                window_size = shown_size
            key = cv2.waitKeyEx(0)
            if key in {ord("q"), 27}:
                write_csv(output, rows, fields)
                print(f"Progresso salvo: {output}")
                return 0
            if key in TAGS:
                tags = [v for v in (row.get("visual_cause_tags") or "").split(";") if v]
                tag = TAGS[key]
                if tag in tags:
                    tags.remove(tag)
                else:
                    tags.append(tag)
                row["visual_cause_tags"] = ";".join(sorted(tags))
                write_csv(output, rows, fields)
                continue
            if key in {8, 127}:
                cursor = max(0, cursor - 1)
                continue
            if key in {ord("r"), ord("R")}:
                row["visual_cause_tags"] = ""
                row["visual_evidence_note"] = ""
                row["visual_review_status"] = "PENDING"
                write_csv(output, rows, fields)
                continue
            if key in {10, 13}:
                row["visual_review_status"] = "COMPLETED"
                write_csv(output, rows, fields)
                cursor += 1
    finally:
        cv2.destroyAllWindows()
    completed = sum(row.get("visual_review_status") == "COMPLETED" for row in rows)
    print(f"Auditoria visual: {completed}/{len(rows)} | {output}")
    return 0


def main() -> int:
    experiment = ROOT / "experiments/ball/BALL-POSITIVE-MISS-AUDIT-EXP-0002"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=experiment / "positive_miss_mechanism_v2.csv")
    parser.add_argument("--output", type=Path, default=experiment / "positive_miss_mechanism_v2_annotated.csv")
    parser.add_argument("--max-preview-width", type=int, default=1600)
    parser.add_argument("--max-preview-height", type=int, default=800)
    parser.add_argument("--window-title", default="Auditoria visual de misses BALL - TRAIN")
    args = parser.parse_args()
    source, fields = read_csv(args.input)
    rows, fields = merge(source, fields, args.output)
    print(f"Misses: {len(rows)}; revisados: {sum(r.get('visual_review_status') == 'COMPLETED' for r in rows)}; pendentes: {sum(r.get('visual_review_status') != 'COMPLETED' for r in rows)}")
    print("Enter salva e avanca. Q salva e sai. Marque somente fatores observaveis; as flags geometricas ficam separadas.")
    return annotate(rows, fields, args.output, args.max_preview_width, args.max_preview_height, args.window_title)


if __name__ == "__main__":
    raise SystemExit(main())
