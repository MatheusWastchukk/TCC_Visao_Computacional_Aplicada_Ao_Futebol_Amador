"""Executa o baseline inicial do detector da bola."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def patch_directml_loss() -> None:
    """Desvia a contagem auxiliar não implementada pelo DirectML para a CPU."""
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.ops import xywh2xyxy

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.to("cpu").unique(return_counts=True)
        counts = counts.to(device=self.device, dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    v8DetectionLoss.preprocess = preprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina o baseline BALL-EXP-0001.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Desativa precisão mista automática, útil para validar compatibilidade ROCm.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device
    if str(args.device).lower() == "directml":
        import torch
        import torch_directml

        # O DirectML atual não aceita inference_mode no NMS do Ultralytics.
        torch.inference_mode = lambda mode=True: torch.no_grad()
        device = torch_directml.device()
    from ultralytics import YOLO
    if str(args.device).lower() == "directml":
        patch_directml_loss()
        import ultralytics.engine.trainer as trainer_module

        trainer_module.select_device = lambda *_args, **_kwargs: device

    started = time.perf_counter()
    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        val=str(args.device).lower() != "directml",
        amp=(str(args.device).lower() != "directml") and not args.no_amp,
        verbose=False,
    )
    metrics = dict(results.results_dict or {})
    summary = {
        "experiment_id": args.output.name,
        "model": str(args.model),
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metrics": {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "execution_seconds": round(time.perf_counter() - started, 3),
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
