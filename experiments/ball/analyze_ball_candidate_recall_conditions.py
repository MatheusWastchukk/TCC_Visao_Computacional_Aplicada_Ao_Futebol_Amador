"""Agrupa o candidate recall de bola por condições observáveis da anotação.

Este utilitario nao executa inferencia. Ele usa o CSV por quadro produzido por
``evaluate_ball_candidate_recall.py`` e acrescenta apenas atributos que podem
ser derivados de forma determinística da anotação: visibilidade, dimensões da
caixa, partida e domínio de captura.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


SIZE_BINS = (
    (16.0, "MUITO_PEQUENA_ATE_16PX"),
    (24.0, "PEQUENA_17_A_24PX"),
    (32.0, "MEDIA_25_A_32PX"),
    (float("inf"), "GRANDE_ACIMA_32PX"),
)


def tamanho_da_bola(largura: float, altura: float) -> str:
    """Classifica a bola pelo maior lado da caixa anotada, em pixels."""
    maior_lado = max(largura, altura)
    for limite, rotulo in SIZE_BINS:
        if maior_lado <= limite:
            return rotulo
    raise AssertionError("Faixas de tamanho incompletas.")


def ler_csv(caminho: Path) -> list[dict[str, str]]:
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def escrever_csv(caminho: Path, linhas: list[dict[str, object]], campos: list[str]) -> None:
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def percentual(numerador: int, denominador: int) -> float:
    return numerador / denominador if denominador else 0.0


def inteiro_booleano(valor: str) -> int:
    """Converte as serializações booleanas aceitas nos CSVs de métricas."""
    normalizado = valor.strip().lower()
    if normalizado in {"1", "true"}:
        return 1
    if normalizado in {"0", "false"}:
        return 0
    raise ValueError(f"Valor booleano inválido: {valor!r}")


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrupa candidate recall por condições observáveis da anotação.",
    )
    parser.add_argument("--annotations", type=Path, required=True, help="CSV ball_dataset_annotations_v2.csv.")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV candidate_recall_frames.csv.")
    parser.add_argument("--output", type=Path, required=True, help="Diretório para os CSVs de auditoria.")
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    anotacoes = ler_csv(args.annotations)
    predicoes = ler_csv(args.predictions)

    obrigatorios_anotacao = {"sample_id", "split", "ball_visibility", "ball_x1", "ball_y1", "ball_x2", "ball_y2", "match_id", "capture_domain"}
    obrigatorios_predicao = {"confidence", "sample_id", "candidate_count", "candidate_hit", "top3_hit"}
    if not anotacoes or not obrigatorios_anotacao.issubset(anotacoes[0]):
        raise ValueError("CSV de anotações não contém todas as colunas necessárias.")
    if not predicoes or not obrigatorios_predicao.issubset(predicoes[0]):
        raise ValueError("CSV de predições não contém todas as colunas necessárias.")

    por_id: dict[str, dict[str, str]] = {}
    for anotacao in anotacoes:
        if anotacao["sample_id"] in por_id:
            raise ValueError(f"sample_id duplicado nas anotações: {anotacao['sample_id']}")
        por_id[anotacao["sample_id"]] = anotacao

    vistos: set[tuple[str, str]] = set()
    linhas_quadro: list[dict[str, object]] = []
    for predicao in predicoes:
        chave = (predicao["confidence"], predicao["sample_id"])
        if chave in vistos:
            raise ValueError(f"Predição duplicada para confidence/sample_id: {chave}")
        vistos.add(chave)
        anotacao = por_id.get(predicao["sample_id"])
        if anotacao is None:
            raise ValueError(f"Predição sem anotação correspondente: {predicao['sample_id']}")
        if anotacao["split"].strip().lower() != "validation":
            raise ValueError(f"Predição fora de validation: {predicao['sample_id']}")
        if anotacao["ball_visibility"] not in {"VISIBLE", "PARTIAL"}:
            raise ValueError(f"Predição para quadro sem bola avaliável: {predicao['sample_id']}")

        largura = float(anotacao["ball_x2"]) - float(anotacao["ball_x1"])
        altura = float(anotacao["ball_y2"]) - float(anotacao["ball_y1"])
        if largura <= 0 or altura <= 0:
            raise ValueError(f"Caixa inválida: {predicao['sample_id']}")
        linhas_quadro.append(
            {
                "confidence": float(predicao["confidence"]),
                "sample_id": predicao["sample_id"],
                "match_id": anotacao["match_id"],
                "capture_domain": anotacao["capture_domain"],
                "ball_visibility": anotacao["ball_visibility"],
                "ball_width_px": round(largura, 3),
                "ball_height_px": round(altura, 3),
                "ball_max_side_px": round(max(largura, altura), 3),
                "ball_area_px": round(largura * altura, 3),
                "size_condition": tamanho_da_bola(largura, altura),
                "candidate_count": int(predicao["candidate_count"]),
                "candidate_hit": inteiro_booleano(predicao["candidate_hit"]),
                "top3_hit": inteiro_booleano(predicao["top3_hit"]),
            }
        )

    grupos: dict[tuple[float, str, str], list[dict[str, object]]] = defaultdict(list)
    for linha in linhas_quadro:
        valor_conf = float(linha["confidence"])
        grupos[(valor_conf, "TOTAL", "TODOS")].append(linha)
        for coluna, tipo in (
            ("ball_visibility", "VISIBILIDADE"),
            ("size_condition", "TAMANHO_CAIXA"),
            ("match_id", "PARTIDA"),
            ("capture_domain", "DOMINIO_CAPTURA"),
        ):
            grupos[(valor_conf, tipo, str(linha[coluna]))].append(linha)

    resumo: list[dict[str, object]] = []
    for (confianca, tipo, condicao), linhas in sorted(grupos.items()):
        total = len(linhas)
        hits = sum(int(linha["candidate_hit"]) for linha in linhas)
        top3 = sum(int(linha["top3_hit"]) for linha in linhas)
        resumo.append(
            {
                "confidence": confianca,
                "group_type": tipo,
                "condition": condicao,
                "frame_count": total,
                "candidate_hits": hits,
                "candidate_recall": round(percentual(hits, total), 6),
                "top3_hits": top3,
                "top3_recall": round(percentual(top3, total), 6),
                "mean_candidates_per_frame": round(sum(int(linha["candidate_count"]) for linha in linhas) / total, 6),
                "median_ball_max_side_px": round(median(float(linha["ball_max_side_px"]) for linha in linhas), 3),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    escrever_csv(
        args.output / "condition_frame_metrics.csv",
        linhas_quadro,
        list(linhas_quadro[0]),
    )
    escrever_csv(
        args.output / "condition_summary.csv",
        resumo,
        list(resumo[0]),
    )
    metadados = {
        "purpose": "Auditoria de candidate recall YOLO por condições observáveis da anotação.",
        "annotations": str(args.annotations),
        "predictions": str(args.predictions),
        "evaluated_frames": len({str(linha["sample_id"]) for linha in linhas_quadro}),
        "prediction_rows": len(linhas_quadro),
        "size_bins_max_side_px": [
            "<=16", "17-24", "25-32", ">32"
        ],
        "not_derived_conditions": [
            "proximidade_de_jogador", "bola_no_ar", "proximidade_do_gol", "desfoque",
        ],
    }
    (args.output / "condition_audit_metadata.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Auditoria concluída: {len(linhas_quadro)} linhas de previsão; {len(resumo)} grupos.")


if __name__ == "__main__":
    main()
