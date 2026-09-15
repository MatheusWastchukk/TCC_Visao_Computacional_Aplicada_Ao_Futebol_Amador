# Configuração congelada V3 — recuperação e redução de eventos candidatos

**Data:** 13/09/2026
**Treino:** somente `TRAIN`
**Confirmação:** uma vez em `VALIDATION`
**TEST:** fechado para avaliação final única

> **Status superado em 13/09/2026:** esta configuração documenta o melhor ponto
> de baixo custo encontrado até então (top 12), mas não é o freeze metodológico
> definitivo. A caracterização posterior mostrou que a arquitetura atual alcança
> 90% de cobertura em `VALIDATION` com top 60 e janela 20+40. O novo objetivo é
> reduzir esse pool amplo preservando cobertura; TEST continua fechado.

## Pipeline operacional

1. Minerar relações geométricas bola × estrutura do gol.
2. Agrupar hits temporais e aplicar o Random Forest temporal congelado.
3. Pré-selecionar até 20 candidatos temporalmente diversos por partida.
4. Calcular o score visual já congelado e aplicar a fusão temporal + visual.
5. Retornar os 12 primeiros candidatos da fusão.
6. Gerar para cada candidato um clipe de 45 s: 15 s antes e 30 s depois do
   timestamp central.

O resultado é uma lista ordenada de `POSSIBLE_GOAL`. O score não confirma gol,
não define `GOAL` automaticamente e não substitui revisão humana.

## Seleção e confirmação

A escolha de top 20 → fusão → top 12 e janela 15+30 foi feita em TRAIN antes da
confirmação externa. Em TRAIN/LOMO, essa configuração cobriu 49/152 gols
(32,24%) em nove minutos de clipes por partida. A confirmação única em
VALIDATION cobriu 18/51 gols (35,29%).

Referências históricas na mesma validação:

| Configuração | Recall@12 |
|---|---:|
| Random Forest temporal, clipes 10+20 | 13/51 (25,49%) |
| Fusão, clipes 10+20 | 17/51 (33,33%) |
| **Fusão, clipes 15+30 (V3)** | **18/51 (35,29%)** |

## Regras após o freeze

Não alterar detector, agrupamento, features, pesos da fusão, quantidade de
pré-candidatos, top 12 ou janela 15+30 usando dados de VALIDATION ou TEST.

Antes de TEST, são permitidos somente trabalhos de engenharia que preservem este
contrato: `analyze_match()`, geração de CSV/JSON, extração de clipes, interface,
testes e verificações de reprodutibilidade.

TEST será executado uma única vez com esta configuração. Seu resultado será
aceito como avaliação final, sem retreino, ajuste de limiar ou nova seleção.
