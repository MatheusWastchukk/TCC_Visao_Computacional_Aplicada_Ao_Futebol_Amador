"""Anota a bola verdadeira nos frames do BALL-PROPOSAL-QUALITY-AUDIT, com retomada."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2"
TAGS = {
    "1": "ground", "2": "aerial", "3": "occlusion", "4": "post_shot",
    "5": "near_goal_or_goalkeeper", "6": "camera_motion", "7": "cross",
    "8": "rebound", "9": "multiple_ball_proposals",
}
FIELDS = ["sample_id", "match_id", "video_filename", "source_stratum", "source_id", "sequence_id",
          "timestamp_seconds", "frame_index", "annotation_state", "ball_visibility",
          "gt_x1", "gt_y1", "gt_x2", "gt_y2", "context_tags"]


def write_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class BallProposalAnnotator:
    def __init__(self, items: list[dict[str, str]], review: list[dict[str, str]], run_dir: Path,
                 review_path: Path, show_completed: bool) -> None:
        self.items, self.review, self.run_dir, self.review_path = items, review, run_dir, review_path
        self.show_completed = show_completed
        self.index = 0
        self.start = None
        self.draft = None
        self.dragging = False
        self.scale = 1.0
        self.image_top = 112
        self.view_size = (1, 1)
        self.original_size = (1, 1)
        self.zoom_center = None
        self.current_crop = (0, 0, 1, 1)
        self.window = "Auditoria de propostas BALL | S visivel, N ausente, U incerto, Q sair"

    def current_tags(self) -> set[str]:
        return {x for x in self.review[self.index].get("context_tags", "").split(";") if x}

    def mouse(self, event, x, y, flags, _param):
        image_x, image_y = x, y - self.image_top
        max_x, max_y = self.view_size
        inside = 0 <= image_x < max_x and 0 <= image_y < max_y
        if event == cv2.EVENT_RBUTTONDOWN and inside:
            crop_x, crop_y, _, _ = self.current_crop
            self.zoom_center = (crop_x + image_x / self.scale, crop_y + image_y / self.scale)
            self.draft = None
            return
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self.start = (image_x, image_y)
            self.dragging = True
            self.draft = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            x1, y1 = self.start
            self.draft = self.display_box((x1, y1, image_x, image_y))
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            x1, y1 = self.start
            self.draft = self.display_box((x1, y1, image_x, image_y))
            self.dragging = False

    def display_box(self, box):
        x1, y1, x2, y2 = box
        sx1, sx2 = sorted((x1, x2))
        sy1, sy2 = sorted((y1, y2))
        if sx2 - sx1 < 3 or sy2 - sy1 < 3:
            return None
        original_w, original_h = self.original_size
        crop_x, crop_y, _, _ = self.current_crop
        return (max(0, min(original_w-1, int(crop_x + sx1 / self.scale))), max(0, min(original_h-1, int(crop_y + sy1 / self.scale))),
                max(1, min(original_w, int(crop_x + sx2 / self.scale))), max(1, min(original_h, int(crop_y + sy2 / self.scale))))

    def pending_indices(self) -> list[int]:
        if self.show_completed:
            return list(range(len(self.items)))
        return [i for i, row in enumerate(self.review) if row.get("annotation_state") != "complete"]

    def advance(self, direction: int) -> bool:
        pending = self.pending_indices()
        if not pending:
            return False
        if self.index in pending:
            pos = pending.index(self.index)
            target = pos + direction
        else:
            later = [p for p in pending if p > self.index]
            earlier = [p for p in pending if p < self.index]
            self.index = (later[0] if later and direction > 0 else
                          earlier[-1] if earlier and direction < 0 else pending[0])
            self.draft = None
            return True
        if 0 <= target < len(pending):
            self.index = pending[target]
            self.draft = None
            return True
        return False

    def save(self, visibility: str) -> None:
        row = self.review[self.index]
        if visibility == "BALL_VISIBLE":
            if self.draft is None:
                print("Desenhe primeiro uma caixa apertada em volta da bola verdadeira.")
                return
            row.update({"ball_visibility": visibility, "gt_x1": str(self.draft[0]), "gt_y1": str(self.draft[1]),
                        "gt_x2": str(self.draft[2]), "gt_y2": str(self.draft[3])})
        else:
            row.update({"ball_visibility": visibility, "gt_x1": "", "gt_y1": "", "gt_x2": "", "gt_y2": ""})
        row["annotation_state"] = "complete"
        row["context_tags"] = ";".join(sorted(self.current_tags()))
        write_atomic(self.review_path, self.review)
        completed = sum(r.get("annotation_state") == "complete" for r in self.review)
        print(f"Salvo: {row['sample_id']} ({completed}/{len(self.review)}) — {visibility}")
        self.advance(1)

    def draw(self):
        item, review = self.items[self.index], self.review[self.index]
        preview = cv2.imread(str(self.run_dir / "overlays_v1" / f"{item['sample_id']}.jpg"))
        if preview is None:
            raise FileNotFoundError(self.run_dir / "overlays_v1" / f"{item['sample_id']}.jpg")
        height, width = preview.shape[:2]
        self.original_size = (width, height)
        if self.zoom_center is None:
            crop_x, crop_y, crop_w, crop_h = 0, 0, width, height
        else:
            crop_w, crop_h = max(1, width // 2), max(1, height // 2)
            crop_x = max(0, min(width-crop_w, int(self.zoom_center[0] - crop_w/2)))
            crop_y = max(0, min(height-crop_h, int(self.zoom_center[1] - crop_h/2)))
        self.current_crop = (crop_x, crop_y, crop_x+crop_w, crop_y+crop_h)
        crop = preview[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        self.scale = min(1240.0 / crop_w, 700.0 / crop_h)
        resized = cv2.resize(crop, (max(1, round(crop_w * self.scale)), max(1, round(crop_h * self.scale))))
        self.view_size = (resized.shape[1], resized.shape[0])
        canvas = __import__("numpy").zeros((self.image_top + resized.shape[0], resized.shape[1], 3), dtype="uint8")
        item_id = item["source_id"]
        text1 = f"{self.index+1}/{len(self.items)}  {item['sample_id']}  {item['match_id']}  {item['source_stratum']}"
        text2 = f"{item_id}  t={item['timestamp_seconds']}s  seq={item.get('sequence_id','')}  frame={item['frame_index']}"
        text3 = "Esq: desenhe GT | Dir: zoom 2x | R: zoom normal | S visivel | N ausente | U incerto | 1-9 tags | Space pula | A/D | Q"
        for y, text in ((24, text1), (52, text2), (82, text3)):
            cv2.putText(canvas, text[:150], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        selected_tags = self.current_tags()
        tag_line = "  ".join(f"{k}:{v}{'*' if v in selected_tags else ''}" for k, v in TAGS.items())
        cv2.putText(canvas, tag_line[:150], (8, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
        canvas[self.image_top:, :resized.shape[1]] = resized
        gt = self.draft
        if gt is None and review.get("ball_visibility") == "BALL_VISIBLE" and review.get("gt_x1"):
            gt = tuple(int(float(review[k])) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2"))
        if gt:
            crop_x, crop_y, _, _ = self.current_crop
            x1, y1, x2, y2 = (int(round((gt[0]-crop_x)*self.scale)), int(round((gt[1]-crop_y)*self.scale)),
                              int(round((gt[2]-crop_x)*self.scale)), int(round((gt[3]-crop_y)*self.scale)))
            cv2.rectangle(canvas, (x1, self.image_top+y1), (x2, self.image_top+y2), (0, 255, 0), 2)
            cv2.putText(canvas, "GT MANUAL", (x1, max(self.image_top+14, self.image_top+y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return canvas

    def run(self) -> None:
        pending = self.pending_indices()
        if not pending:
            print("Todos os frames estão anotados.")
            return
        self.index = pending[0]
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse)
        try:
            while True:
                if not self.pending_indices():
                    print("Anotacao concluida para todos os frames.")
                    break
                cv2.imshow(self.window, self.draw())
                key = cv2.waitKey(30) & 0xFF
                if key == 255:
                    continue
                char = chr(key).lower() if key < 128 else ""
                if char == "q":
                    write_atomic(self.review_path, self.review)
                    break
                if char == "s":
                    self.save("BALL_VISIBLE")
                elif char == "n":
                    self.save("BALL_NOT_VISIBLE")
                elif char == "u":
                    self.save("UNCERTAIN")
                elif char == "d":
                    self.advance(1)
                elif char == "a":
                    self.advance(-1)
                elif key == 32:
                    self.advance(1)
                elif key == 8:
                    self.draft = None
                elif char == "r":
                    self.zoom_center = None
                    self.draft = None
                elif char in TAGS:
                    row = self.review[self.index]
                    tags = self.current_tags()
                    tag = TAGS[char]
                    tags.remove(tag) if tag in tags else tags.add(tag)
                    row["context_tags"] = ";".join(sorted(tags))
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--review-file", type=Path, default=None)
    parser.add_argument("--show-completed", action="store_true", help="Allow navigation through completed frames.")
    args = parser.parse_args()
    manifest = list(csv.DictReader((args.run_dir / "sample_manifest_v1.csv").open(newline="", encoding="utf-8-sig")))
    if not manifest or any(row["split"] != "train" for row in manifest):
        raise ValueError("Expected a non-empty TRAIN-only sample manifest.")
    if not (args.run_dir / "inference_progress_v1.json").is_file() or \
       json_load(args.run_dir / "inference_progress_v1.json").get("status") != "completed":
        raise RuntimeError("GPU/CPU proposal inference must complete before annotation.")
    review_path = args.review_file or args.run_dir / "manual_ball_review_v1.csv"
    old = {}
    if review_path.exists():
        old = {row["sample_id"]: row for row in csv.DictReader(review_path.open(newline="", encoding="utf-8-sig"))}
    review = []
    for item in manifest:
        review.append({field: old.get(item["sample_id"], {}).get(field, default) for field, default in {
            "sample_id": item["sample_id"], "match_id": item["match_id"], "video_filename": item["video_filename"],
            "source_stratum": item["source_stratum"], "source_id": item["source_id"], "sequence_id": item.get("sequence_id", ""),
            "timestamp_seconds": item["timestamp_seconds"], "frame_index": item["frame_index"],
            "annotation_state": "pending", "ball_visibility": "", "gt_x1": "", "gt_y1": "", "gt_x2": "", "gt_y2": "", "context_tags": ""}.items()})
    write_atomic(review_path, review)
    BallProposalAnnotator(manifest, review, args.run_dir, review_path, args.show_completed).run()
    return 0


def json_load(path: Path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
