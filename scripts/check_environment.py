"""Check the local development environment for the TCC project."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "yaml": "pyyaml",
}


@dataclass(frozen=True)
class ModuleCheck:
    import_name: str
    package_name: str
    available: bool
    version: str | None = None
    error: str | None = None


def _module_version(module: object) -> str | None:
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def check_module(import_name: str, package_name: str) -> ModuleCheck:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # pragma: no cover - exact import errors vary by machine.
        return ModuleCheck(import_name, package_name, False, error=str(exc))

    return ModuleCheck(import_name, package_name, True, version=_module_version(module))


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None

    try:
        result = subprocess.run(
            [executable, "-version" if command == "ffmpeg" else "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return executable

    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0] if first_line else executable


def collect_environment() -> dict[str, Any]:
    start = perf_counter()
    module_checks = [
        check_module(import_name, package_name)
        for import_name, package_name in REQUIRED_MODULES.items()
    ]

    torch_info: dict[str, object] = {}
    if any(check.import_name == "torch" and check.available for check in module_checks):
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    return {
        "project_root": str(Path(__file__).resolve().parents[1]),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tools": {
            "git": _command_version("git"),
            "ffmpeg": _command_version("ffmpeg"),
        },
        "modules": [asdict(check) for check in module_checks],
        "torch": torch_info,
        "elapsed_seconds": round(perf_counter() - start, 3),
    }


def main() -> int:
    environment = collect_environment()
    print(json.dumps(environment, indent=2, ensure_ascii=False))

    missing_modules = [
        item["package_name"] for item in environment["modules"] if not item["available"]
    ]
    missing_tools = [
        name for name, version in environment["tools"].items() if version is None
    ]

    if missing_modules or missing_tools:
        print()
        print("Environment check failed.")
        if missing_modules:
            print("Missing Python packages:", ", ".join(missing_modules))
        if missing_tools:
            print("Missing system tools:", ", ".join(missing_tools))
        return 1

    print()
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
