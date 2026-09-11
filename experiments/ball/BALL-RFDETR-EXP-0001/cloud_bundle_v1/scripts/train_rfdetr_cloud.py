"""Treina RF-DETR Small em NVIDIA para o teste controlado de nuvem."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina RF-DETR Small no dataset COCO incluído no pacote.")
    parser.add_argument("--dataset", type=Path, required=True, help="Diretório COCO com train e valid.")
    parser.add_argument("--output", type=Path, required=True, help="Diretório persistente dos checkpoints.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Desativa AMP; use somente se houver erro CUDA.")
    return parser.parse_args()


def main() -> int:
    args = argumentos()
    if not (args.dataset / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"Dataset COCO de treino ausente: {args.dataset}")
    if not (args.dataset / "valid" / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"Dataset COCO de validação ausente: {args.dataset}")

    import torch
    from rfdetr import RFDETRSmall

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível. Verifique a instância NVIDIA e o PyTorch instalado.")

    args.output.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    model = RFDETRSmall(
        num_classes=1,
        resolution=args.resolution,
        amp=not args.no_amp,
        device=args.device,
    )
    model.train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        resolution=args.resolution,
        device=args.device,
        amp=not args.no_amp,
        notes={
            "experiment_id": "BALL-RFDETR-EXP-0001-CLOUD-NVIDIA",
            "protocol": "Mesmo dataset e split de desenvolvimento do teste local; test fechado não utilizado.",
        },
    )
    manifesto = {
        "status": "completed",
        "execution_seconds": round(time.perf_counter() - inicio, 3),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "resolution": args.resolution,
        "device": args.device,
        "amp": not args.no_amp,
    }
    (args.output / "cloud_training_manifest.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifesto, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
