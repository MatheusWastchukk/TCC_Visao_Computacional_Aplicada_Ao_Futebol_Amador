"""Extrai sequências compactas de frames para clipes do verifier, somente TRAIN."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai frames de clipes TRAIN para o verifier de eventos.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=640)
    args = parser.parse_args()
    if args.frames_per_clip < 2 or args.max_width < 1:
        raise ValueError("frames-per-clip deve ser ao menos 2 e max-width deve ser positivo.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos: {args.output}.")
    with args.manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if any(row["split"] != "train" for row in rows):
        raise AssertionError("A extração aceita somente TRAIN; TEST permanece fechado.")
    args.output.mkdir(parents=True, exist_ok=False)
    index_rows = []
    for item_index, row in enumerate(rows, start=1):
        capture = cv2.VideoCapture(str(args.videos_dir / row["video_filename"]))
        if not capture.isOpened():
            raise FileNotFoundError(args.videos_dir / row["video_filename"])
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            start, end = float(row["clip_start_seconds"]), float(row["clip_end_seconds"])
            for frame_order in range(args.frames_per_clip):
                timestamp = start + (end - start) * frame_order / (args.frames_per_clip - 1)
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Frame indisponível em {row['verifier_example_id']} @ {timestamp:.3f}")
                if image.shape[1] > args.max_width:
                    scale = args.max_width / image.shape[1]
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                path = args.output / f"{row['verifier_example_id']}_{frame_order:02d}.jpg"
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"Falha ao gravar {path}")
                index_rows.append({"verifier_example_id": row["verifier_example_id"], "frame_order": frame_order, "timestamp_seconds": f"{timestamp:.3f}", "frame_path": str(path), "label": row["label"], "sampling_role": row["sampling_role"]})
        finally:
            capture.release()
        print(f"Concluído {item_index}/{len(rows)}: {row['verifier_example_id']}", flush=True)
    fields = list(index_rows[0])
    with (args.output / "frame_index_v1.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(index_rows)
    summary = {"experiment_id": "GOAL-EVENT-VERIFIER-FRAMES-V1", "scope": "somente TRAIN", "examples": len(rows), "frames": len(index_rows), "frames_per_clip": args.frames_per_clip, "max_width": args.max_width}
    (args.output / "summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
