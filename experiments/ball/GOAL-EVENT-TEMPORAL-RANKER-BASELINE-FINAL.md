# GOAL-EVENT-TEMPORAL-RANKER-BASELINE-FINAL

## Configuração congelada de desenvolvimento

- Escopo: somente `TRAIN`.
- Grupos: hits de bola × estrutura, agrupados com gap máximo de 3 s e mínimo de 2 hits.
- Rótulo fraco: melhor hit a até ±10 s de timestamp de gol auditado.
- Features: 20 agregadas e 16 em quatro blocos temporais normalizados.
- Modelo: `RandomForestClassifier` com 300 árvores, `min_samples_leaf=2` e balanceamento por classe.
- Protocolo: leave-one-match-out.
- Seleção: até quatro candidatos por partida, separados por ao menos 45 s.
- Janela de clipe: 10 s antes e 20 s depois do melhor hit.

## Evidência LOMO

| Recall@K | Cobertura temporal |
|---|---:|
| @4 | 18/152 (11,84%) |
| @8 | 30/152 (19,74%) |
| @12 | 36/152 (23,68%) |
| @20 | 50/152 (32,89%) |

Esses valores medem cobertura temporal de desenvolvimento em `TRAIN`, não precisão, F1, desempenho online ou resultado final.

## Decisão

Não realizar novas varreduras de features ou detectores nesta fase. A próxima etapa é o verificador de clipes, mantendo `VALIDATION` sem ajustes e `TEST` fechado.
