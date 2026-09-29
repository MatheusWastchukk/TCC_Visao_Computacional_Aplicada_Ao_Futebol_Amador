"""Exporta segmentos operacionais V4 a partir de um pool já pontuado de uma partida."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib

PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.events.operational_modes_v4 import MODES, analyze_match
from src.events.temporal_inference_v4 import select_wide_pool, temporal_groups
from src.events.visual_fusion_inference_v4 import fusion_scores, visual_scores


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


ProgressCallback = Callable[[str, float], None]


def load_script_main(path: Path) -> Callable[..., int]:
    """Carrega um estágio Python também quando a aplicação está empacotada."""
    spec = spec_from_file_location("goal_event_miner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar o estágio: {path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def main(argv: list[str] | None = None, progress: ProgressCallback | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera segmentos V4 a partir de um pool pontuado ou de uma partida TRAIN.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-scores", type=Path, help="CSV de até 60 candidatos com escores temporal e de fusão.")
    source.add_argument("--video-path", type=Path, help="Vídeo de uma partida TRAIN ou vídeo externo autorizado.")
    parser.add_argument("--match-id", help="Identificador do manifesto ou da partida externa; obrigatório com --video-path.")
    parser.add_argument("--allow-arbitrary-video", action="store_true", help="Permite vídeo externo na aplicação desktop.")
    parser.add_argument("--split-manifest", type=Path, default=Path("experiments/datasets/dataset_split_v1.csv"))
    parser.add_argument("--videos-dir", type=Path, default=Path("data/Videos_Edited"))
    parser.add_argument("--goal-model", type=Path, default=Path("runs/detect/experiments/goal_geometry/EXP-0092/yolo11n_cpu_reference/weights/best.pt"))
    parser.add_argument("--ball-model", type=Path, default=Path("runs/detect/experiments/ball/BALL-EXP-0005/weights/best.pt"))
    parser.add_argument("--temporal-ranker", type=Path, default=Path("experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_ranker.joblib"))
    parser.add_argument("--visual-verifier", type=Path, default=Path("experiments/ball/GOAL-EVENT-PIPELINE-FINAL-V1/temporal_sequence_verifier.joblib"))
    parser.add_argument("--fusion-model", type=Path, default=Path("experiments/ball/GOAL-EVENT-FUSION-FINAL-V1/fusion_reranker.joblib"))
    parser.add_argument("--ball-imgsz", type=int, default=960, help="Resolução da busca BALL global durante mineração.")
    parser.add_argument("--roi-imgsz", type=int, default=None, help="Ativa a ablação de crop BALL em alta resolução.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", help="Retoma após mineração concluída sem repetir os hits já gerados.")
    parser.add_argument("--mode", choices=sorted(MODES), default="cobertura_reduzida")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"A saída já existe: {args.output}")
    if args.resume and (not args.output.is_dir() or (args.output / "results.json").exists()):
        raise ValueError("A retomada exige uma saída parcial sem results.json.")
    if args.candidate_scores:
        candidates = read_csv(args.candidate_scores)
    else:
        if not args.match_id:
            raise ValueError("--match-id é obrigatório com --video-path.")
        if not args.video_path.is_file():
            raise FileNotFoundError(args.video_path)
        if not args.allow_arbitrary_video:
            manifest = read_csv(args.split_manifest)
            rows = [row for row in manifest if row["match_id"] == args.match_id]
            if len(rows) != 1 or rows[0]["split"] != "train":
                raise AssertionError("A inferência integrada aceita somente uma partida TRAIN; VALIDATION e TEST permanecem fechados.")
            expected_video = (args.videos_dir / rows[0]["video_filename"]).resolve()
            if args.video_path.resolve() != expected_video:
                raise ValueError("O vídeo deve corresponder exatamente à partida TRAIN do manifesto.")
        for path in (args.goal_model, args.ball_model, args.temporal_ranker, args.visual_verifier, args.fusion_model):
            if not path.is_file():
                raise FileNotFoundError(f"Artefato congelado ausente: {path}")
        miner_output = args.output / "stages" / "mining"
        frames_path = miner_output / "candidate_frames_v1.csv"
        if not (args.resume and frames_path.is_file()):
            if progress:
                progress("Mineração de candidatos", 0)
            mine_main = load_script_main(PROJECT_ROOT / "experiments/ball/mine_goal_event_candidates_v1.py")
            mine_main(
                ["--split-manifest", str(args.split_manifest), "--videos-dir", str(args.videos_dir),
                 "--goal-model", str(args.goal_model), "--ball-model", str(args.ball_model),
                 "--include-matches", args.match_id,
                 "--expected-split", "external" if args.allow_arbitrary_video else "train",
                 "--ball-imgsz", str(args.ball_imgsz),
                 "--max-candidates-per-match", "60", "--output", str(miner_output),
                 "--device", str(args.device)]
                + (["--roi-imgsz", str(args.roi_imgsz)] if args.roi_imgsz is not None else [])
                + (["--video-path", str(args.video_path), "--video-id", args.match_id] if args.allow_arbitrary_video else []),
                progress,
            )
        if progress:
            progress("Ranking temporal e fusão", 70)
        groups = temporal_groups(read_csv(miner_output / "candidate_frames_v1.csv"))
        if not groups:
            raise RuntimeError("A mineração não produziu grupos temporais elegíveis.")
        temporal = select_wide_pool(groups, joblib.load(args.temporal_ranker))
        visual = visual_scores(temporal, args.videos_dir, args.visual_verifier)
        candidates = fusion_scores(visual, args.fusion_model)
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / "wide_pool_scored.csv", candidates, list(candidates[0]))
    if progress:
        progress("União dos segmentos", 82)
    result = analyze_match(candidates, args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = result["selected_candidates"]
    assert isinstance(selected, list)
    write_csv(args.output / "selected_candidates.csv", selected, list(selected[0]) if selected else ["event_id"])
    segments = result["segments"]
    assert isinstance(segments, list)
    rows = [{**segment, "member_event_ids": ";".join(segment["member_event_ids"])} for segment in segments]
    fields = ["segment_id", "label", "start_seconds", "end_seconds", "start_hhmmss", "end_hhmmss", "duration_seconds", "representative_event_id", "representative_timestamp_seconds", "representative_time_hhmmss", "rank_score", "fusion_score_goal", "member_event_ids", "member_count"]
    write_csv(args.output / "segments.csv", rows, fields)
    if progress:
        progress("Segmentos prontos", 85)
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
