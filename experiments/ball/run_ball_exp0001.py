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
    from ultralytics.utils.tal import TaskAlignedAssigner
    from ultralytics.utils.ops import xywh2xyxy

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.to("cpu").unique(return_counts=True)
        counts = counts.to(device=self.device, dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        # DirectML currently lacks scatter_add_ for this CPU-created index
        # bookkeeping. Compute per-image positions on CPU, then move only the
        # compact index tensor back to the model device.
        within_cpu: list[int] = []
        seen = [0] * batch_size
        for image_idx in batch_idx.detach().to("cpu").tolist():
            within_cpu.append(seen[int(image_idx)])
            seen[int(image_idx)] += 1
        within_idx = torch.tensor(within_cpu, dtype=torch.long, device=self.device)
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    v8DetectionLoss.preprocess = preprocess

    original_select_topk = TaskAlignedAssigner.select_topk_candidates

    def select_topk_candidates(self, metrics: torch.Tensor, topk_mask: torch.Tensor | None = None) -> torch.Tensor:
        # The DirectML backend rejects scatter_add_ when indices and the
        # destination differ outside the scatter dimension. This assigner is
        # under no_grad, so keep its small count operation on CPU and return
        # the mask to the model device.
        if metrics.device.type == "privateuseone":
            topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
            if topk_mask is None:
                topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
            topk_idxs = topk_idxs.masked_fill(~topk_mask, 0).to("cpu")
            count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device="cpu")
            count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
            count_tensor.masked_fill_(count_tensor > 1, 0)
            return count_tensor.to(device=metrics.device, dtype=metrics.dtype)
        return original_select_topk(self, metrics, topk_mask)

    TaskAlignedAssigner.select_topk_candidates = select_topk_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina o baseline BALL-EXP-0001.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-val", action="store_true", help="N\u00e3o executa valida\u00e7\u00e3o durante o treino.")
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
        # Ultralytics assumes non-CPU devices expose the CUDA memory API.
        # DirectML has no memory_reserved(), and memory reporting is optional.
        trainer_module.BaseTrainer._get_memory = lambda self, fraction=False: 0.0

    started = time.perf_counter()
    model = YOLO(str(args.model))
    validation_enabled = (str(args.device).lower() != "directml") and not args.no_val
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
        val=validation_enabled,
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
        "metrics_scope": (
            "configured validation split"
            if validation_enabled
            else "post-training training-fold diagnostics only; not an independent holdout or a promotion metric"
        ),
        "metrics": {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "execution_seconds": round(time.perf_counter() - started, 3),
        "status": "completed",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
