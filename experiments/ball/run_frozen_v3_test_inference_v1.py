"""Executa uma única inferência sem rótulos para as políticas V3 congeladas em TEST."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[2]
BALL = ROOT / "experiments/ball"
sys.path.insert(0, str(BALL))

import evaluate_goal_clip_pretrained_verifier_lomo_v1 as clip_reader  # noqa: E402
import evaluate_nested_time_budget_lomo_fullpool_v1 as policy  # noqa: E402
import extract_net_motion_features_v1 as net_features  # noqa: E402
import mine_goal_event_candidates_v1 as miner  # noqa: E402

SPLIT_MANIFEST = ROOT / "experiments/datasets/dataset_split_v1.csv"
VIDEOS_DIR = ROOT / "data/Videos_Test_Fechado"
FREEZE_MANIFEST = ROOT / "experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v3.json"
MODE_DIR = ROOT / "experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3"
MODE_MANIFEST = MODE_DIR / "model_manifest_v3.json"
GOAL_MODEL = ROOT / "runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt"
BALL_MODEL = ROOT / "runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"
FIELDS = [
    "hit_count", "unique_timestamps", "duration_seconds", "max_gap_seconds", "gap_count",
    "max_candidate_score", "mean_candidate_score", "max_ball_confidence", "mean_ball_confidence",
    "max_goal_confidence", "mean_goal_confidence", "min_normalized_distance", "mean_normalized_distance",
    "distance_std", "distance_before_min", "distance_after_min", "distance_slope", "inside_goal_fraction",
    "mean_relative_speed", "max_relative_speed",
    *[f"bin_{i}_{n}" for i in range(4) for n in ("mean_distance", "mean_score", "mean_ball_confidence", "inside_fraction")],
]
DERIVED = [
    "trajectory_distance_delta_0_3", "trajectory_score_delta_0_3", "trajectory_ball_confidence_delta_0_3",
    "trajectory_inside_delta_0_3", "trajectory_distance_abs_change_sum", "trajectory_score_abs_change_sum",
    "trajectory_distance_decrease_fraction", "trajectory_score_increase_fraction", "trajectory_distance_slope",
    "trajectory_score_slope", "trajectory_ball_confidence_slope", "trajectory_distance_min_bin",
    "trajectory_score_max_bin",
]
NET_COLUMNS = ["net_net_motion_inside_mean", "net_net_motion_ring_mean", "net_net_motion_ratio"]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Recusando CSV vazio: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"Metadados inválidos de vídeo: {path}")
    return float(count / fps)


def feature_rows(frame_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups = net_features.temporal_groups(frame_rows, 3.0, 2)
    group_indexes: dict[str, int] = defaultdict(int)
    output: list[dict[str, Any]] = []
    for group in groups:
        first = group[0]
        match_id = first["match_id"]
        group_indexes[match_id] += 1
        uid = f"{match_id}::temporal_group_{group_indexes[match_id]:04d}"
        times = np.asarray([float(row["timestamp_seconds"]) for row in group], dtype=np.float64)
        distances = np.asarray([float(row["normalized_distance"]) for row in group], dtype=np.float64)
        scores = np.asarray([float(row["candidate_score"]) for row in group], dtype=np.float64)
        balls = np.asarray([float(row["ball_confidence"]) for row in group], dtype=np.float64)
        goals = np.asarray([float(row["goal_confidence"]) for row in group], dtype=np.float64)
        unique_times = np.unique(times)
        gaps = np.diff(unique_times)
        min_index = int(np.argmin(distances))
        speeds: list[float] = []
        for previous, current in zip(group, group[1:]):
            dt = float(current["timestamp_seconds"]) - float(previous["timestamp_seconds"])
            if dt > 0:
                p = json.loads(previous["ball_box_json"])
                c = json.loads(current["ball_box_json"])
                speeds.append(math.hypot((float(c[0]) + float(c[2]) - float(p[0]) - float(p[2])) / 2,
                                         (float(c[1]) + float(c[3]) - float(p[1]) - float(p[3])) / 2) / dt)
        row: dict[str, Any] = {
            "group_uid": uid, "match_id": match_id, "video_filename": first["video_filename"],
            "best_timestamp_seconds": f"{times[int(np.argmax(scores))]:.6f}",
            "hit_count": len(group), "unique_timestamps": len(unique_times),
            "duration_seconds": float(times[-1] - times[0]),
            "max_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
            "gap_count": len(gaps), "max_candidate_score": float(scores.max()),
            "mean_candidate_score": float(scores.mean()), "max_ball_confidence": float(balls.max()),
            "mean_ball_confidence": float(balls.mean()), "max_goal_confidence": float(goals.max()),
            "mean_goal_confidence": float(goals.mean()), "min_normalized_distance": float(distances.min()),
            "mean_normalized_distance": float(distances.mean()), "distance_std": float(distances.std()),
            "distance_before_min": float(distances[:min_index].mean()) if min_index else float(distances[min_index]),
            "distance_after_min": float(distances[min_index + 1:].mean()) if min_index + 1 < len(distances) else float(distances[min_index]),
            "distance_slope": float(np.polyfit(times - times[0], distances, 1)[0]) if len(unique_times) > 1 else 0.0,
            "inside_goal_fraction": float(np.mean([row["ball_inside_expanded_goal"] == "True" for row in group])),
            "mean_relative_speed": float(np.mean(speeds)) if speeds else 0.0,
            "max_relative_speed": float(np.max(speeds)) if speeds else 0.0,
        }
        relative = (times - times[0]) / (times[-1] - times[0]) if times[-1] > times[0] else np.zeros_like(times)
        for bin_index in range(4):
            lower, upper = bin_index / 4, (bin_index + 1) / 4
            mask = (relative >= lower) & (relative < upper if bin_index < 3 else relative <= upper)
            row[f"bin_{bin_index}_mean_distance"] = float(distances[mask].mean()) if mask.any() else 0.0
            row[f"bin_{bin_index}_mean_score"] = float(scores[mask].mean()) if mask.any() else 0.0
            row[f"bin_{bin_index}_mean_ball_confidence"] = float(balls[mask].mean()) if mask.any() else 0.0
            row[f"bin_{bin_index}_inside_fraction"] = float(np.mean([group[i]["ball_inside_expanded_goal"] == "True" for i in np.where(mask)[0]])) if mask.any() else 0.0
        d = np.asarray([row[f"bin_{i}_mean_distance"] for i in range(4)])
        s = np.asarray([row[f"bin_{i}_mean_score"] for i in range(4)])
        b = np.asarray([row[f"bin_{i}_mean_ball_confidence"] for i in range(4)])
        inside = np.asarray([row[f"bin_{i}_inside_fraction"] for i in range(4)])
        row.update({
            "trajectory_distance_delta_0_3": d[3] - d[0], "trajectory_score_delta_0_3": s[3] - s[0],
            "trajectory_ball_confidence_delta_0_3": b[3] - b[0], "trajectory_inside_delta_0_3": inside[3] - inside[0],
            "trajectory_distance_abs_change_sum": float(np.abs(np.diff(d)).sum()),
            "trajectory_score_abs_change_sum": float(np.abs(np.diff(s)).sum()),
            "trajectory_distance_decrease_fraction": float(np.mean(np.diff(d) <= 0)),
            "trajectory_score_increase_fraction": float(np.mean(np.diff(s) >= 0)),
            "trajectory_distance_slope": float(np.polyfit(np.arange(4), d, 1)[0]),
            "trajectory_score_slope": float(np.polyfit(np.arange(4), s, 1)[0]),
            "trajectory_ball_confidence_slope": float(np.polyfit(np.arange(4), b, 1)[0]),
            "trajectory_distance_min_bin": float(np.argmin(d) / 3), "trajectory_score_max_bin": float(np.argmax(s) / 3),
        })
        for name in (*FIELDS, *DERIVED):
            row[name] = float(f"{float(row[name]):.6f}")
        output.append(row)
    return output


def bounded_clip(start: float, end: float, duration_seconds: float) -> tuple[float, float]:
    width = end - start
    if duration_seconds <= width:
        return 0.0, duration_seconds
    bounded_start = min(max(0.0, start), duration_seconds - width)
    return bounded_start, bounded_start + width


def merged_intervals(rows: list[dict[str, Any]], duration_seconds: float) -> list[tuple[float, float]]:
    intervals = sorted((max(0.0, float(r["delivery_start_seconds"])), min(duration_seconds, float(r["delivery_end_seconds"]))) for r in rows)
    merged: list[list[float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--reuse-mining", action="store_true", help="Reutiliza mineração TEST completa já existente na pasta de saída.")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.reuse_mining:
        raise FileExistsError(f"A pasta de saída não está vazia: {output}")
    if not FREEZE_MANIFEST.is_file() or not MODE_MANIFEST.is_file():
        raise FileNotFoundError("Manifestos do freeze V3 ausentes.")
    freeze = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if freeze.get("manifest_version") != 3 or freeze.get("is_final_sprint_freeze") is not True:
        raise RuntimeError("Freeze final V3 não confirmado.")
    mode_protocol = freeze["frozen_test_protocol"]["modes"]
    expected_modes = {"FAST_V3", "BALANCED_V3", "HIGH_RECALL_V3"}
    if set(mode_protocol) != expected_modes:
        raise RuntimeError(f"Políticas do freeze inesperadas: {sorted(mode_protocol)}")
    model_manifest = json.loads(MODE_MANIFEST.read_text(encoding="utf-8"))
    for name, item in model_manifest.items():
        model_path = ROOT / item["path"]
        if not model_path.is_file() or model_path.stat().st_size != item["bytes"] or sha256(model_path) != item["sha256"]:
            raise RuntimeError(f"Hash/tamanho do modelo V3 divergente: {name}")
    for path in (SPLIT_MANIFEST, GOAL_MODEL, BALL_MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    split_rows, _ = read_csv(SPLIT_MANIFEST)
    test_rows = [row for row in split_rows if row.get("split", "").lower() == "test"]
    if len(test_rows) != 9 or len({row["match_id"] for row in test_rows}) != 9:
        raise RuntimeError("O manifesto precisa conter exatamente nove partidas TEST únicas.")
    for row in test_rows:
        if not (VIDEOS_DIR / row["video_filename"]).is_file():
            raise FileNotFoundError(VIDEOS_DIR / row["video_filename"])

    output.mkdir(parents=True, exist_ok=True)
    mining_dir = output / "mining_test_v1"
    miner_args = [
        "--split-manifest", str(SPLIT_MANIFEST), "--videos-dir", str(VIDEOS_DIR),
        "--goal-model", str(GOAL_MODEL), "--ball-model", str(BALL_MODEL),
        "--include-matches", *[row["match_id"] for row in test_rows], "--expected-split", "test",
        "--analysis-fps", "5", "--goal-confidence", "0.25", "--ball-confidence", "0.05",
        "--goal-imgsz", "640", "--ball-imgsz", "960", "--goal-margin", "0.35",
        "--max-normalized-distance", "2.0", "--merge-gap-seconds", "3", "--minimum-hits", "2",
        "--min-separation-seconds", "45", "--max-candidates-per-match", "60",
        "--clip-pre-seconds", "10", "--clip-post-seconds", "20", "--device", args.device,
        "--output", str(mining_dir),
    ]
    mining_metrics_path = mining_dir / "metrics.json"
    if args.reuse_mining:
        if not mining_metrics_path.is_file() or not (mining_dir / "candidate_frames_v1.csv").is_file():
            raise FileNotFoundError("Não há uma mineração completa para reutilizar.")
        mining_metrics = json.loads(mining_metrics_path.read_text(encoding="utf-8"))
        if mining_metrics.get("videos_processed") != len(test_rows):
            raise RuntimeError("A mineração existente não cobre as nove partidas TEST.")
        mining_parameters = mining_metrics.get("parameters", {})
        if mining_parameters.get("expected_split") != "test" or mining_parameters.get("device") != args.device:
            raise RuntimeError("Os parâmetros da mineração existente não correspondem a esta execução.")
        mining_wall = sum(float(item.get("elapsed_seconds", 0.0)) for item in mining_metrics.get("by_match", {}).values())
    else:
        started = time.perf_counter()
        miner.main(miner_args)
        mining_wall = time.perf_counter() - started
    frame_path = mining_dir / "candidate_frames_v1.csv"
    frame_rows, frame_fields = read_csv(frame_path)
    if not frame_rows or {row["match_id"] for row in frame_rows} != {row["match_id"] for row in test_rows}:
        raise RuntimeError("O pool de frames não cobre exatamente as nove partidas TEST.")

    temporal_started = time.perf_counter()
    features = feature_rows(frame_rows)
    temporal_feature_seconds = time.perf_counter() - temporal_started
    if not features:
        raise RuntimeError("Nenhum grupo temporal elegível foi gerado em TEST.")
    durations = {r["match_id"]: duration(VIDEOS_DIR / r["video_filename"]) for r in test_rows}
    net_dir = output / "net_motion_features_test_v1"
    net_started = time.perf_counter()
    subprocess.run([
        sys.executable, str(BALL / "extract_net_motion_features_v1.py"),
        "--frames", str(frame_path), "--videos-dir", str(VIDEOS_DIR),
        "--output", str(net_dir), "--frame-gap-seconds", "0.2",
    ], check=True)
    net_motion_seconds = time.perf_counter() - net_started
    net_rows, _ = read_csv(net_dir / "net_motion_groups_v1.csv")
    net_by_uid = {row["group_uid"]: row for row in net_rows}
    if set(net_by_uid) != {row["group_uid"] for row in features}:
        raise RuntimeError("Os grupos das features net-motion não correspondem ao pool temporal.")

    models = {}
    model_names = {
        "temporal_ranker": "temporal_ranker.joblib", "net_motion_ranker": "net_motion_ranker.joblib",
        "visual_verifier": "visual_verifier.joblib", "fusion_ranker": "fusion_ranker.joblib",
    }
    for name, filename in model_names.items():
        models[name] = joblib.load(ROOT / model_manifest[name]["path"])
    temporal_artifact, net_artifact = models["temporal_ranker"], models["net_motion_ranker"]
    temporal_names = temporal_artifact["features"]
    net_names = net_artifact["features"]
    required = set(temporal_names) | set(net_names)
    if required - (set(features[0]) | set(NET_COLUMNS)):
        raise RuntimeError(f"Atributos TEST não cobrem o contrato V3: {sorted(required - (set(features[0]) | set(NET_COLUMNS)))}")

    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    weights = ResNet18_Weights.DEFAULT
    resnet_checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not resnet_checkpoint.is_file():
        raise FileNotFoundError(f"Pesos do ResNet18 não estão em cache: {resnet_checkpoint}")
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval().to(device)
    transform = weights.transforms()
    visual_embeddings: list[np.ndarray] = []
    visual_seconds_by_match: dict[str, float] = defaultdict(float)
    for index, row in enumerate(features, start=1):
        visual_started = time.perf_counter()
        match = row["match_id"]
        center = float(row["best_timestamp_seconds"])
        start, end = bounded_clip(center - 10.0, center + 20.0, durations[match])
        frames = clip_reader.read_clip_frames(VIDEOS_DIR / row["video_filename"], start, end, 8)
        batch = torch.stack([transform(frame) for frame in frames]).to(device)
        with torch.inference_mode():
            visual_embeddings.append(encoder(batch).detach().cpu().numpy().astype(np.float32).reshape(-1))
        visual_seconds_by_match[match] += time.perf_counter() - visual_started
        if index % 100 == 0 or index == len(features):
            print(f"Embeddings visuais TEST: {index}/{len(features)}", flush=True)
    embedding_matrix = np.stack(visual_embeddings)
    embedding_path = output / "candidate_embeddings_test_v1.npy"
    np.save(embedding_path, embedding_matrix, allow_pickle=False)
    embedding_ids_path = output / "candidate_embedding_ids_test_v1.json"
    embedding_ids_path.write_text(json.dumps([r["group_uid"] for r in features], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    temporal_x = np.asarray([[float(row[name]) for name in temporal_names] for row in features], dtype=np.float32)
    # Join raw net-motion fields under the names used by the frozen training matrix.
    for row in features:
        nf = net_by_uid[row["group_uid"]]
        row["net_net_motion_inside_mean"] = nf["net_motion_inside_mean"]
        row["net_net_motion_ring_mean"] = nf["net_motion_ring_mean"]
        row["net_net_motion_ratio"] = nf["net_motion_ratio"]
    net_x = np.asarray([[float(row[name]) for name in net_names] for row in features], dtype=np.float32)
    scoring_started = time.perf_counter()
    temporal_model = temporal_artifact["model"]
    net_model = net_artifact["model"]
    temporal_scores = temporal_model.predict_proba(temporal_x)[:, list(temporal_model.classes_).index(1)]
    net_scores = net_model.predict_proba(net_x)[:, list(net_model.classes_).index(1)]
    visual_model = models["visual_verifier"]
    visual_scores = visual_model.predict_proba(embedding_matrix)[:, list(visual_model.classes_).index(1)]
    fusion_artifact = models["fusion_ranker"]
    fusion_x = np.asarray([[float(temporal_scores[i]), float(visual_scores[i])] for i in range(len(features))], dtype=np.float64)
    fusion_model = fusion_artifact["model"]
    fusion_scores = fusion_model.predict_proba(fusion_x)[:, list(fusion_model.classes_).index(1)]
    scoring_seconds = time.perf_counter() - scoring_started

    score_rows: list[dict[str, Any]] = []
    for i, row in enumerate(features):
        center = float(row["best_timestamp_seconds"])
        clip_start, clip_end = bounded_clip(center - 10.0, center + 20.0, durations[row["match_id"]])
        score_rows.append({
            "candidate_id": row["group_uid"], "group_uid": row["group_uid"], "match_id": row["match_id"],
            "video_filename": row["video_filename"], "split": "test", "best_timestamp_seconds": f"{center:.6f}",
            "candidate_start_seconds": f"{clip_start:.6f}", "candidate_end_seconds": f"{clip_end:.6f}",
            "delivery_start_seconds": f"{center - 20.0:.6f}", "delivery_end_seconds": f"{center + 40.0:.6f}",
            "rank_score": f"{float(temporal_scores[i]):.8f}", "visual_score_goal": f"{float(visual_scores[i]):.8f}",
            "fusion_score_goal": f"{float(fusion_scores[i]):.8f}", "net_motion": f"{float(net_scores[i]):.8f}",
        })
    write_csv(output / "ranked_candidates_test_v1.csv", score_rows)

    selection_started = time.perf_counter()
    per_match = []
    selected_rows: list[dict[str, Any]] = []
    for test_row in test_rows:
        match = test_row["match_id"]
        source = [row for row in score_rows if row["match_id"] == match]
        match_summary: dict[str, Any] = {
            "match_id": match, "video_filename": test_row["video_filename"], "duration_seconds": durations[match],
            "candidate_groups": len(source), "miner_elapsed_seconds": miner_summary(mining_dir, match),
            "net_motion_stage_seconds": net_motion_seconds * len(source) / len(score_rows),
            "visual_embedding_seconds": visual_seconds_by_match[match],
            "ranking_stage_seconds": scoring_seconds * len(source) / len(score_rows),
        }
        for mode_name, mode in mode_protocol.items():
            delivered = policy.delivery_rows(source, int(mode["window_seconds"]))
            selected = policy.select_policy(delivered, str(mode["method"]), int(mode["k_per_match"]))
            merged = merged_intervals(selected, durations[match])
            unique_seconds = sum(end - start for start, end in merged)
            match_summary[f"{mode_name}_selected_candidates"] = len(selected)
            match_summary[f"{mode_name}_segments"] = len(merged)
            match_summary[f"{mode_name}_unique_minutes"] = unique_seconds / 60.0
            match_summary[f"{mode_name}_video_percent"] = 100.0 * unique_seconds / durations[match]
            match_summary[f"{mode_name}_budget_minutes"] = float(mode["budget_minutes_per_match"])
            match_summary[f"{mode_name}_budget_exceeded"] = unique_seconds / 60.0 > float(mode["budget_minutes_per_match"]) + 1e-9
            for row in selected:
                selected_rows.append({
                    **row, "mode": mode_name,
                    "delivery_start_seconds": f"{float(row['delivery_start_seconds']):.6f}",
                    "delivery_end_seconds": f"{float(row['delivery_end_seconds']):.6f}",
                })
        per_match.append(match_summary)
    write_csv(output / "selected_candidates_test_v1.csv", selected_rows)
    write_csv(output / "per_match_inference_test_v1.csv", per_match)
    selection_seconds = time.perf_counter() - selection_started

    miner_metrics = json.loads((mining_dir / "metrics.json").read_text(encoding="utf-8"))
    source_paths = [
        Path(__file__), Path(miner.__file__), Path(net_features.__file__), Path(policy.__file__), Path(clip_reader.__file__),
        BALL / "extract_local_trajectory_features_v1.py", ROOT / "src/events/net_motion_v1.py",
        BALL / "prepare_goal_event_temporal_dataset_v1.py", BALL / "prepare_goal_event_rich_temporal_dataset_v1.py",
        FREEZE_MANIFEST, MODE_MANIFEST, SPLIT_MANIFEST, GOAL_MODEL, BALL_MODEL,
    ]
    model_paths = [ROOT / item["path"] for item in model_manifest.values()]
    report = {
        "status": "TEST_V3_INFERENCE_COMPLETE",
        "scope": "TEST9, três políticas fixas V3; ranking não consultou timestamps ou rótulos TEST",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "test", "matches": len(test_rows), "candidate_groups": len(score_rows),
        "policies": {name: {k: v[k] for k in ("budget_minutes_per_match", "method", "window_seconds", "k_per_match", "gap_seconds", "rescue_enabled")} for name, v in mode_protocol.items()},
        "parameters": miner_metrics["parameters"],
        "per_match_mining": miner_metrics["by_match"],
        "mining_wall_seconds": mining_wall,
        "stage_runtime_seconds": {
            "mining_wall": mining_wall, "temporal_feature_generation": temporal_feature_seconds,
            "net_motion_extraction": net_motion_seconds,
            "visual_embedding_extraction": sum(visual_seconds_by_match.values()),
            "model_scoring": scoring_seconds, "fixed_policy_selection_and_unions": selection_seconds,
        },
        "device": args.device, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "resnet_weights_enum": str(weights),
        "resnet_checkpoint_sha256": sha256(resnet_checkpoint),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths if p.is_file()},
        "model_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in model_paths},
        "detector_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (GOAL_MODEL, BALL_MODEL)},
        "embedding_sha256": sha256(embedding_path),
        "features_sha256": sha256(output / "ranked_candidates_test_v1.csv"),
        "split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "feature_groups": len(features), "catalog_loaded_by_inference": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch_threads": args.torch_threads},
    }
    (output / "inference_manifest_test_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "matches": report["matches"], "candidate_groups": report["candidate_groups"], "output": str(output)}, ensure_ascii=False, indent=2), flush=True)
    return 0


def miner_summary_path(output: Path) -> Path:
    return output / "metrics.json"


def miner_summary(output: Path, match_id: str) -> float | None:
    data = json.loads(miner_summary_path(output).read_text(encoding="utf-8"))
    value = data.get("by_match", {}).get(match_id, {}).get("elapsed_seconds")
    return float(value) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
