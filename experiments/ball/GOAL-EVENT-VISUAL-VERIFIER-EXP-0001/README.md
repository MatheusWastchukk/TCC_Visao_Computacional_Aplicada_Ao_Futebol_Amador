# GOAL-EVENT-VISUAL-VERIFIER-EXP-0001

Baseline visual de desenvolvimento para verificar clipes candidatos a gol.

## Escopo e protocolo

- Somente `TRAIN`; `VALIDATION` e `TEST` não foram abertos.
- Avaliação leave-one-match-out em 100 clipes da revisão original do minerador: 10 `GOAL` e 90 `NO_GOAL`.
- Os 36 positivos confirmados por alinhamento prévio de timestamp não participaram da avaliação primária, pois essa seleção enviesaria a estimativa.
- Cada clipe é descrito por oito frames a 640 px, com estatísticas de iluminação, saturação, bordas, histogramas HSV e suas variações temporais.

## Resultado

| Modelo | AP | AUC ROC | Recall em 0,5 |
| --- | ---: | ---: | ---: |
| Regressão logística | 9,88% | 45,67% | 30,00% |
| Floresta aleatória | 15,35% | 51,78% | 10,00% |

## Decisão

**NO-GO para o descritor visual artesanal.** O melhor resultado ficou próximo de uma ordenação aleatória e não sustenta seu uso como verificador de gol.

Uma eventual segunda versão deve usar um codificador visual pré-treinado, mas depende de definir hipótese, origem dos pesos e orçamento de execução. Não interpretar estes números como desempenho final ou avaliação independente.

## Artefatos

- `visual_features_v1.csv`: 48 descritores por clipe.
- `predictions_lomo_v1.csv`: escores fora da partida de treino de cada exemplo.
- `summary_v1.json`: métricas consolidadas.
