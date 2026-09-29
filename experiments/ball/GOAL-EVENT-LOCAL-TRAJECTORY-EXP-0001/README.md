# GOAL-EVENT-LOCAL-TRAJECTORY-EXP-0001

## Hipótese

Adicionar quatro atributos derivados de uma trajetória local escolhida por
programação dinâmica poderia reduzir saltos para objetos aleatórios e melhorar
o ranking de candidatos a gol.

## Protocolo

- Escopo exclusivo em `TRAIN`.
- Os atributos foram extraídos de `candidate_frames_v1.csv` já existente; não
  houve nova mineração nem consulta a `VALIDATION` ou `TEST`.
- Foram associados 5.372 grupos de 22 partidas a timestamps catalogados
  concluídos dessas mesmas partidas.
- A avaliação foi leave-one-match-out com o mesmo `RandomForestClassifier`
  do baseline enriquecido: a partida avaliada nunca participou do ajuste.
- Comparação pareada: mesma amostra, mesmas janelas e mesmos limites `K`; a
  única diferença foi a inclusão dos quatro atributos locais.

## Atributos testados

- `local_trajectory_path_observations`;
- `local_trajectory_path_score`;
- `local_trajectory_mean_normalized_step`;
- `local_trajectory_continuity`.

Uma observação isolada recebe continuidade zero, pois não constitui evidência
de trajetória.

## Resultado LOMO em TRAIN

| K | Baseline | Com trajetória local | Diferença |
|---:|---:|---:|---:|
| 4 | 14/130 (10,77%) | 16/130 (12,31%) | +2 |
| 8 | 24/130 (18,46%) | 22/130 (16,92%) | -2 |
| 12 | 31/130 (23,85%) | 28/130 (21,54%) | -3 |
| 20 | 48/130 (36,92%) | 45/130 (34,62%) | -3 |
| 30 | 58/130 (44,62%) | 57/130 (43,85%) | -1 |
| 60 | 85/130 (65,38%) | 84/130 (64,62%) | -1 |

## Decisão

**NO-GO.** O ganho isolado em `K=4` não compensa as perdas em todos os demais
limites avaliados. A evidência não será integrada ao V4, não será confirmada
em `VALIDATION` e não altera a aplicação. O próximo experimento deve investigar
evidência de chute/ação ofensiva, conforme a prioridade registrada após a
conversa com o orientador.

Estes números são somente desenvolvimento em `TRAIN`; não são avaliação geral
nem resultado final do TCC.
