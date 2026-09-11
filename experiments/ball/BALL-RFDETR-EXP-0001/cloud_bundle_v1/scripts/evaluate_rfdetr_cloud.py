"""Avalia candidate recall e candidatas em NOT_VISIBLE do checkpoint RF-DETR."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def iou(caixa_a: tuple[float, float, float, float], caixa_b: tuple[float, float, float, float]) -> float:
    """Calcula IoU entre duas caixas xyxy."""
    x1, y1 = max(caixa_a[0], caixa_b[0]), max(caixa_a[1], caixa_b[1])
    x2, y2 = min(caixa_a[2], caixa_b[2]), min(caixa_a[3], caixa_b[3])
    intersecao = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, caixa_a[2] - caixa_a[0]) * max(0.0, caixa_a[3] - caixa_a[1])
    area_b = max(0.0, caixa_b[2] - caixa_b[0]) * max(0.0, caixa_b[3] - caixa_b[1])
    uniao = area_a + area_b - intersecao
    return intersecao / uniao if uniao else 0.0


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia checkpoint RF-DETR no split validation incluído no pacote.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Diretório dataset_rfdetr/valid/images.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="BALL-RFDETR-EXP-0001-CLOUD-NVIDIA")
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def condicao_tamanho(largura: float, altura: float) -> str:
    """Classifica pelo maior lado da caixa anotada, nas faixas locais."""
    maior_lado = max(largura, altura)
    if maior_lado <= 16:
        return "MUITO_PEQUENA_ATE_16PX"
    if maior_lado <= 24:
        return "PEQUENA_17_A_24PX"
    if maior_lado <= 32:
        return "MEDIA_25_A_32PX"
    return "GRANDE_ACIMA_32PX"


def ler_csv(caminho: Path) -> list[dict[str, str]]:
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def salvar_csv(caminho: Path, linhas: list[dict[str, object]]) -> None:
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def main() -> int:
    args = argumentos()
    from rfdetr import RFDETRSmall

    anotacoes = [
        linha for linha in ler_csv(args.annotations)
        if linha["split"].strip().lower() == "validation"
        and linha["ball_visibility"] in {"VISIBLE", "PARTIAL", "NOT_VISIBLE"}
    ]
    if len(anotacoes) != 119:
        raise ValueError(f"Esperadas 119 anotações de validation; encontradas {len(anotacoes)}.")
    model = RFDETRSmall.from_checkpoint(args.checkpoint, device=args.device)
    linhas: list[dict[str, object]] = []
    for confianca in args.confidences:
        for indice, anotacao in enumerate(anotacoes, start=1):
            # A exportação COCO padroniza o nome como <sample_id>.jpg; o CSV
            # de origem preserva o timestamp no nome do frame e não pode ser
            # usado diretamente para localizar a cópia do pacote.
            imagem = args.images / f"{anotacao['sample_id']}.jpg"
            if not imagem.exists():
                raise FileNotFoundError(f"Imagem ausente no pacote: {imagem}")
            deteccoes = model.predict(str(imagem), threshold=confianca, include_source_image=False)
            caixas = [tuple(float(valor) for valor in caixa) for caixa in deteccoes.xyxy.tolist()]
            visibilidade = anotacao["ball_visibility"]
            positivo = visibilidade in {"VISIBLE", "PARTIAL"}
            if positivo:
                gt = tuple(float(anotacao[campo]) for campo in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"))
                largura, altura = gt[2] - gt[0], gt[3] - gt[1]
                sobreposicoes = [iou(gt, caixa) for caixa in caixas]
                correspondentes = [posicao for posicao, valor in enumerate(sobreposicoes) if valor >= args.iou]
                rank = correspondentes[0] + 1 if correspondentes else 0
                acerto = bool(correspondentes)
                melhor_iou = max(sobreposicoes, default=0.0)
                tamanho = condicao_tamanho(largura, altura)
                maior_lado = max(largura, altura)
            else:
                rank, acerto, melhor_iou = 0, False, 0.0
                tamanho, maior_lado = "", ""
            linhas.append(
                {
                    "model_id": args.model_id,
                    "confidence": f"{confianca:.2f}",
                    "sample_id": anotacao["sample_id"],
                    "match_id": anotacao["match_id"],
                    "ball_visibility": visibilidade,
                    "evaluation_type": "POSITIVE" if positivo else "NOT_VISIBLE",
                    "size_condition": tamanho,
                    "ball_max_side_px": maior_lado,
                    "candidate_count": len(caixas),
                    "best_iou": f"{melhor_iou:.6f}",
                    "matching_rank": rank,
                    "candidate_hit": acerto,
                    "top3_hit": bool(rank and rank <= 3),
                }
            )
            if indice % 25 == 0 or indice == len(anotacoes):
                print(f"conf={confianca:.2f} processados={indice}/{len(anotacoes)}")

    grupos: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for linha in linhas:
        confianca = str(linha["confidence"])
        if linha["evaluation_type"] == "POSITIVE":
            grupos[(confianca, "positive_total", "VISIBLE_PARTIAL")].append(linha)
            grupos[(confianca, "visibility", str(linha["ball_visibility"]))].append(linha)
            grupos[(confianca, "size_condition", str(linha["size_condition"]))].append(linha)
        else:
            grupos[(confianca, "not_visible", "NOT_VISIBLE")].append(linha)

    resumo: list[dict[str, object]] = []
    for (confianca, tipo, condicao), grupo in sorted(grupos.items()):
        positivos = [linha for linha in grupo if linha["evaluation_type"] == "POSITIVE"]
        total = len(grupo)
        acertos = sum(bool(linha["candidate_hit"]) for linha in positivos)
        top3 = sum(bool(linha["top3_hit"]) for linha in positivos)
        resumo.append(
            {
                "model_id": args.model_id,
                "confidence": confianca,
                "group_type": tipo,
                "group_value": condicao,
                "frames": total,
                "candidate_hits": acertos if positivos else "",
                "candidate_recall": f"{acertos / len(positivos):.6f}" if positivos else "",
                "top3_recall": f"{top3 / len(positivos):.6f}" if positivos else "",
                "mean_candidate_count": f"{sum(int(linha['candidate_count']) for linha in grupo) / total:.6f}",
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    salvar_csv(args.output / "candidate_recall_frames.csv", linhas)
    salvar_csv(args.output / "candidate_recall_summary.csv", resumo)
    print(f"Avaliação concluída: {len(linhas)} linhas; saída={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
