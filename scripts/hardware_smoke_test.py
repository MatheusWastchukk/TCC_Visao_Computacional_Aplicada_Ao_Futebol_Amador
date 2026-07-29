"""Run a small hardware smoke test over exactly 100 video frames."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import psutil
import torch

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_ARTIFACT_DIR = Path("experiments/hardware-spike/artifacts")


def find_first_clip() -> Path | None:
    for folder in (Path("data/clips"), Path("data/raw")):
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                return path
    return None


def create_synthetic_video(path: Path, frame_count: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps = 1280, 720, 30.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video at {path}")

    for frame_index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = 110
        frame[:, :, 0] = 35
        center_x = 80 + (frame_index * 11) % (width - 160)
        center_y = 120 + (frame_index * 5) % (height - 240)
        cv2.circle(frame, (center_x, center_y), 12, (245, 245, 245), -1)
        cv2.rectangle(frame, (width - 170, 240), (width - 35, 480), (255, 255, 255), 3)
        cv2.putText(
            frame,
            f"frame {frame_index + 1:03d}",
            (32, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def resolve_video(input_video: str | None) -> tuple[Path, bool]:
    if input_video:
        return Path(input_video), False

    existing_clip = find_first_clip()
    if existing_clip is not None:
        return existing_clip, False

    synthetic_path = DEFAULT_ARTIFACT_DIR / "synthetic_100_frames.mp4"
    create_synthetic_video(synthetic_path)
    return synthetic_path, True


def resolve_torch_device(device_name: str) -> Any:
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device 'cuda' requested, but CUDA is not available")
        return torch.device("cuda")

    if device_name == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            message = "Device 'dml' requested, but torch-directml is not installed"
            raise RuntimeError(message) from exc
        return torch_directml.device()

    raise RuntimeError(f"Unsupported device: {device_name}")


def process_frames(video_path: Path, frame_limit: int, device_name: str) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Device {device_name!r} requested, but CUDA is not available")

    device = resolve_torch_device(device_name)
    process = psutil.Process()
    memory_before_mb = process.memory_info().rss / (1024 * 1024)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    start = perf_counter()
    frame_count = 0
    checksum = 0.0

    while frame_count < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break

        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).to(device=device, dtype=torch.float32)
        checksum += float(tensor.mean().item())
        frame_count += 1

    elapsed_seconds = perf_counter() - start
    capture.release()

    memory_after_mb = process.memory_info().rss / (1024 * 1024)
    fps = frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return {
        "video": str(video_path),
        "device": device_name,
        "torch_device": str(device),
        "frames_requested": frame_limit,
        "frames_processed": frame_count,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "fps": round(fps, 2),
        "memory_before_mb": round(memory_before_mb, 2),
        "memory_after_mb": round(memory_after_mb, 2),
        "memory_delta_mb": round(memory_after_mb - memory_before_mb, 2),
        "checksum": round(checksum, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"])
    parser.add_argument(
        "--output",
        default="experiments/hardware-spike/cpu_smoke_test.json",
    )
    args = parser.parse_args()

    video_path, synthetic = resolve_video(args.video)
    result = process_frames(video_path, args.frames, args.device)
    result.update(
        {
            "synthetic_video": synthetic,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
