"""Run one BALL frame using workspace-local MIOpen database/cache paths."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
MIOPEN_CACHE_ROOT = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/miopen_workspace_cache_v1"
MIOPEN_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
for cache_subdir in ("user_db", "kernel_cache"):
    (MIOPEN_CACHE_ROOT / cache_subdir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(MIOPEN_CACHE_ROOT / "user_db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(MIOPEN_CACHE_ROOT / "kernel_cache"))

import cv2
import torch
from ultralytics import YOLO

WEIGHTS = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
FRAME = ROOT / "experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2/frames_v1/ball_proposal_0001.jpg"

if not torch.cuda.is_available():
    raise RuntimeError("ROCm device unavailable")
frame = cv2.imread(str(FRAME))
if frame is None:
    raise FileNotFoundError(FRAME)
print({"device": torch.cuda.get_device_name(0),
       "miopen_user_db": os.environ.get("MIOPEN_USER_DB_PATH"),
       "miopen_kernel_cache": os.environ.get("MIOPEN_CUSTOM_CACHE_DIR")}, flush=True)
result = YOLO(str(WEIGHTS)).predict(source=frame, conf=0.05, imgsz=960, device=0, verbose=False)[0]
print({"status": "completed", "ball_proposals": len(result.boxes)}, flush=True)
