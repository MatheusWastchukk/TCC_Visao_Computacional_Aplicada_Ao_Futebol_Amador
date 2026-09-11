#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/output"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/train/checkpoint_best_ema.pth}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA indisponível. Verifique o driver NVIDIA e o PyTorch."
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
PY

python "${ROOT_DIR}/scripts/train_rfdetr_cloud.py" \
  --dataset "${ROOT_DIR}/dataset_rfdetr" \
  --output "${OUTPUT_DIR}/train" \
  --epochs 30 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --learning-rate 0.0001 \
  --resolution 512 \
  --device cuda

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint esperado não encontrado: ${CHECKPOINT}" >&2
  echo "Localize checkpoint_best_ema.pth em ${OUTPUT_DIR}/train e execute apenas a avaliação." >&2
  exit 1
fi

python "${ROOT_DIR}/scripts/evaluate_rfdetr_cloud.py" \
  --checkpoint "${CHECKPOINT}" \
  --annotations "${ROOT_DIR}/metadata/ball_dataset_annotations_v2.csv" \
  --images "${ROOT_DIR}/dataset_rfdetr/valid/images" \
  --output "${OUTPUT_DIR}/candidate_recall" \
  --device cuda
