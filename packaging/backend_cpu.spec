# Pacote preliminar Windows/CPU em pasta, sem dados ou resultados TRAIN/TEST.
from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

os.environ["YOLO_AUTOINSTALL"] = "false"
root = Path(SPECPATH).parent
encoder = Path(os.environ["GOAL_REVIEW_BUILD_RESNET"])
assets = [
    "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt",
    "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt",
    "experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib",
    "experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_sequence_verifier.joblib",
    "experiments/ball/GOAL-EVENT-FUSION-FINAL-V1/fusion_reranker.joblib",
    "experiments/ball/mine_goal_event_candidates_v1.py",
    "src/events/operational_modes_v4.py",
]
datas = [(str(root / name), str(Path(name).parent)) for name in assets]
datas += [(str(encoder), "models"), (str(root / "desktop/node_modules/ffmpeg-static/ffmpeg.exe"), "ffmpeg")]
datas += collect_data_files("ultralytics") + copy_metadata("ultralytics")
hiddenimports = collect_submodules("ultralytics", filter=lambda name: not name.startswith("ultralytics.trackers"))
hiddenimports += collect_submodules("sklearn", filter=lambda name: ".tests" not in name and not name.endswith(".conftest"))
hiddenimports += ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.h11_impl", "uvicorn.lifespan.on"]
binaries = []
a = Analysis(
    [str(root / "backend/launcher.py")], pathex=[str(root)],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide6", "torchaudio", "pytest", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="goal-review-api", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name=os.environ.get("GOAL_REVIEW_BUILD_FOLDER", "goal-review-api"), upx=False)
