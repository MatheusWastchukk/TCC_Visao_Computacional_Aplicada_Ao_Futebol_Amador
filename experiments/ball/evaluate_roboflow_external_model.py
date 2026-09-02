"""Avalia um modelo Roboflow externo no protocolo local de detecção da bola."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from inference_sdk import InferenceConfiguration, InferenceHTTPClient


def get_api_key() -> str:
    """Obtém a chave sem gravá-la nos artefatos do experimento."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as environment:
            value, _ = winreg.QueryValueEx(environment, "ROBOFLOW_API_KEY")
            return str(value).strip()
    except (FileNotFoundError, OSError):
        return ""


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def prediction_box(prediction: dict) -> tuple[float, float, float, float]:
    """Converte a caixa x/y/largura/altura retornada pela API para xyxy."""
    x, y = float(prediction["x"]), float(prediction["y"])
    width, height = float(prediction["width"]), float(prediction["height"])
    return (x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia um modelo Roboflow externo nas imagens locais de validação.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="football-ball-detection-rejhg/4")
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Limita imagens para smoke test; zero avalia todas.")
    args = parser.parse_args()
    api_key = get_api_key()
    if not api_key:
        parser.error("ROBOFLOW_API_KEY não encontrada no ambiente atual nem no ambiente do usuário.")
    with args.annotations.open(newline="", encoding="utf-8") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "validation" and row["ball_visibility"] in {"VISIBLE", "PARTIAL", "NOT_VISIBLE"}
        ]
    if args.limit:
        rows = rows[: args.limit]
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
    client.configure(InferenceConfiguration(confidence_threshold=args.confidence))
    args.output.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        response = client.infer(str(image_path), model_id=args.model_id)
        predictions = response.get("predictions", [])
        candidates = sorted(
            [(prediction_box(prediction), float(prediction.get("confidence", 0.0)), str(prediction.get("class", ""))) for prediction in predictions],
            key=lambda item: item[1], reverse=True,
        )
        best_box, best_confidence, best_class = candidates[0] if candidates else (None, 0.0, "")
        has_ground_truth = row["ball_visibility"] in {"VISIBLE", "PARTIAL"}
        ground_truth = tuple(float(row[field]) for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")) if has_ground_truth else None
        overlap = iou(ground_truth, best_box) if ground_truth and best_box else 0.0
        candidate_ious = [iou(ground_truth, box) for box, _, _ in candidates] if ground_truth else []
        best_candidate_iou = max(candidate_ious, default=0.0)
        matching_rank = next((index + 1 for index, value in enumerate(candidate_ious) if value >= args.iou), 0)
        if has_ground_truth:
            status = "TP" if best_box and overlap >= args.iou else "FN"
        else:
            status = "FP" if best_box else "TN"
        audit_rows.append({
            "sample_id": row["sample_id"], "match_id": row["match_id"], "capture_domain": row["capture_domain"],
            "ball_visibility": row["ball_visibility"], "timestamp_hhmmss": row["timestamp_hhmmss"],
            "status": status, "prediction_count": str(len(candidates)), "prediction_class": best_class,
            "prediction_confidence": f"{best_confidence:.6f}" if best_box else "",
            "prediction_box_json": json.dumps(best_box) if best_box else "",
            "top1_iou": f"{overlap:.6f}", "best_candidate_iou": f"{best_candidate_iou:.6f}",
            "matching_candidate_rank": str(matching_rank), "image_path": str(image_path),
        })
        print(f"[{index}/{len(rows)}] {row['sample_id']} -> {status}")
    metrics = {
        "experiment_id": args.output.name,
        "model_id": args.model_id,
        "source": "Roboflow Serverless Hosted API",
        "images_sent_to_external_api": len(audit_rows),
        "confidence": args.confidence,
        "iou": args.iou,
        "tp": sum(row["status"] == "TP" for row in audit_rows),
        "tn": sum(row["status"] == "TN" for row in audit_rows),
        "fp": sum(row["status"] == "FP" for row in audit_rows),
        "fn": sum(row["status"] == "FN" for row in audit_rows),
        "positive_frames": sum(row["ball_visibility"] in {"VISIBLE", "PARTIAL"} for row in audit_rows),
        "candidate_recall": round(
            sum(int(row["matching_candidate_rank"]) > 0 for row in audit_rows if row["ball_visibility"] in {"VISIBLE", "PARTIAL"})
            / max(1, sum(row["ball_visibility"] in {"VISIBLE", "PARTIAL"} for row in audit_rows)),
            6,
        ),
        "top1_identity_accuracy": round(
            sum(row["status"] == "TP" for row in audit_rows if row["ball_visibility"] in {"VISIBLE", "PARTIAL"})
            / max(1, sum(row["ball_visibility"] in {"VISIBLE", "PARTIAL"} for row in audit_rows)),
            6,
        ),
        "status": "completed",
    }
    with (args.output / "external_prediction_audit.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
