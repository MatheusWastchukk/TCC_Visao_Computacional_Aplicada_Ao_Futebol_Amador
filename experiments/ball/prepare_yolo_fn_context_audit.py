"""Prepara overlays e uma fila para revisar o contexto de falsos negativos YOLO.

Não executa inferência. A classificação de contexto é humana e deve ser feita
depois com ``annotate_ball_error_context.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara a fila de revisão de contexto dos falsos negativos YOLO.",
    )
    parser.add_argument("--annotations", type=Path, required=True, help="CSV de anotações de bola.")
    parser.add_argument("--frame-metrics", type=Path, required=True, help="CSV condition_frame_metrics.csv.")
    parser.add_argument("--output", type=Path, required=True, help="Diretório da fila e dos overlays.")
    parser.add_argument("--confidence", type=float, default=0.05, help="Limiar de confiança a revisar.")
    return parser.parse_args()


def ler_csv(caminho: Path) -> list[dict[str, str]]:
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def salvar_csv(caminho: Path, linhas: list[dict[str, object]]) -> None:
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def desenhar_overlay(imagem, anotacao: dict[str, str], confianca: float) -> object:
    """Desenha a caixa de referência para orientar a inspeção do falso negativo."""
    overlay = imagem.copy()
    x1, y1, x2, y2 = (
        int(float(anotacao[campo]))
        for campo in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")
    )
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 3)
    cv2.putText(overlay, "GT BOLA", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.putText(
        overlay,
        f"FALSO NEGATIVO | conf={confianca:.2f} | IoU alvo >= 0.50",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 255),
        2,
    )
    return overlay


def main() -> int:
    args = argumentos()
    anotacoes = ler_csv(args.annotations)
    metricas = ler_csv(args.frame_metrics)
    por_id = {linha["sample_id"]: linha for linha in anotacoes}
    if len(por_id) != len(anotacoes):
        raise ValueError("Há sample_id duplicado nas anotações.")

    falsos_negativos = [
        linha for linha in metricas
        if abs(float(linha["confidence"]) - args.confidence) < 1e-9
        and linha["candidate_hit"].strip() == "0"
    ]
    if not falsos_negativos:
        raise ValueError("Não há falsos negativos para o limiar solicitado.")

    args.output.mkdir(parents=True, exist_ok=True)
    diretorio_overlays = args.output / "overlays"
    diretorio_overlays.mkdir(exist_ok=True)
    fila: list[dict[str, object]] = []
    for indice, metrica in enumerate(falsos_negativos, start=1):
        anotacao = por_id.get(metrica["sample_id"])
        if anotacao is None:
            raise ValueError(f"Métrica sem anotação: {metrica['sample_id']}")
        caminho_imagem = Path(anotacao["image_path"])
        imagem = cv2.imread(str(caminho_imagem))
        if imagem is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")
        caminho_overlay = diretorio_overlays / f"{indice:03d}_{metrica['sample_id']}.jpg"
        if not cv2.imwrite(str(caminho_overlay), desenhar_overlay(imagem, anotacao, args.confidence)):
            raise RuntimeError(f"Não foi possível salvar o overlay: {caminho_overlay}")
        fila.append(
            {
                "sample_id": metrica["sample_id"],
                "match_id": anotacao["match_id"],
                "capture_domain": anotacao["capture_domain"],
                "timestamp_hhmmss": anotacao["timestamp_hhmmss"],
                "ball_visibility": anotacao["ball_visibility"],
                "size_condition": metrica["size_condition"],
                "ball_max_side_px": metrica["ball_max_side_px"],
                "candidate_count": metrica["candidate_count"],
                "overlay_path": caminho_overlay.as_posix(),
                "review_label": "FALSO_NEGATIVO",
                "error_context": "",
                "context_status": "pending",
                "context_notes": "",
            }
        )

    caminho_fila = args.output / "fn_context_review_queue.csv"
    salvar_csv(caminho_fila, fila)
    print(f"Fila criada: {len(fila)} falsos negativos; CSV={caminho_fila}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
