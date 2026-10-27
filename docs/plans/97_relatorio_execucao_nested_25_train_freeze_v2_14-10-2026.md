# Relatório 97 — Execução Nested 25/25 TRAIN e freeze V2

**Data:** 14/10/2026<br>
**Objetivo:** executar a recomendação para refazer o Nested LOMO na população TRAIN completa antes da avaliação final, sem ampliar o grid.<br>
**Estado:** Nested, consolidação, seleção dos modos e freeze V2 concluídos; avaliação preditiva de TEST pendente de timestamps humanos.

## 1. Resumo da execução

O Nested anterior abrangia 22 partidas e 130 timestamps avaliáveis porque a matriz `net_motion` disponível à época não incluía `match_02`, `match_04` e `match_06`. Foi recuperado o material dessas três partidas, reconstruídos seus 370 grupos ausentes e montada uma matriz full-pool com 5.742 grupos em 25 partidas. A auditoria confirmou alinhamento completo com as features temporais, o pool de scores visuais e os 152 timestamps TRAIN concluídos.

Com a população alinhada, foi executado o Nested LOMO rigoroso em 25 folds, preservando os métodos, os valores de K, as janelas, o gap, os modelos e a regra de orçamento registrados anteriormente. Foram recalculados os resultados por partida e por gol, o bootstrap pareado, a fronteira Pareto e as três configurações fixas dos modos V2. O freeze final V2 foi gravado após o preflight histórico e o preflight adicional do full-pool passarem.

Depois do freeze, foi verificado o split TEST para preparar a avaliação. Existem nove partidas e 57 gols registrados por contagem, mas não há timestamps humanos de TEST no checkout. Os modelos V2 **não foram inferidos sobre TEST**; sem timestamps independentes, não é possível medir recall de gols nem ganhos/perdas por partida. Foi criado um template pendente e um anotador específico para preparar esse catálogo sem mostrar predições ao anotador.

## 2. Reconstrução e auditoria da população TRAIN

| Etapa | Resultado |
|---|---:|
| Registros históricos `net_motion` preservados | 5.372 grupos / 22 partidas |
| Grupos reconstruídos em `match_02`, `match_04`, `match_06` | 370/370 |
| População comum final | 5.742 grupos / 25 partidas |
| Gols catalogados avaliáveis | 152 em 25 partidas |
| Diferenças nos 5.372 grupos históricos auditados | 0 |

Os registros reconstruídos usaram o extrator existente e `frame_gap_seconds=0.2`. Features temporais derivadas, métricas de movimento, rótulos, timestamps e chaves históricas foram comparados com as fontes disponíveis. Os novos conjuntos foram gravados em pastas próprias; os resultados Nested de 22 partidas e a matriz histórica não foram sobrescritos.

O CSV full-pool de `net_motion` tem SHA-256 `d8fb1948d4cc63423bb06631d698c2e4fad74cd0c89773aad132eb23fff3cca2`.

## 3. Protocolo Nested executado

- **Outer folds:** 25, deixando uma partida inteira fora de treino, normalização, fusão e escolha da política em cada fold.
- **População:** 5.742 grupos e 152 gols avaliáveis.
- **Métodos:** `TEMPORAL`, `RRF3`, `RRF4_NET`, `CORE35_FUSION10_NET10` e `CORE40_RRF4_NET_RESCUE`.
- **K:** 20, 30, 40, 50, 55 e 60.
- **Janelas:** 30, 40, 45 e 60 segundos; gap de 45 segundos.
- **Orçamentos:** 30, 40, 45, 50 e 58 minutos únicos por partida.
- **Famílias:** `GLOBAL_BEST`, `NO_RESCUE` e `CORE_RESCUE`, selecionadas separadamente em cada fold e orçamento.
- **Seleção interna:** maximizar gols cobertos entre políticas que respeitam o teto em todas as partidas internas; desempatar por minutos únicos, segmentos e ordem fixa de complexidade.
- **Modelos:** RF de 300 árvores, `min_samples_leaf=2`, `class_weight=balanced`, `random_state=42`; regressões logísticas e embeddings congelados conforme o protocolo existente.

A saída tem 375 linhas de partida×orçamento×família e 2.280 linhas de partida×gol×orçamento×família. Foram confirmadas 25 partidas, 152 gols por orçamento/família, ausência de duplicatas e zero violações do orçamento nos folds externos.

## 4. Estimativas Nested em TRAIN

Cada valor mostra gols cobertos de 152, recall e minutos únicos médios de vídeo por partida.

| Orçamento | GLOBAL_BEST | NO_RESCUE | CORE_RESCUE |
|---:|---:|---:|---:|
| 30 min | 104/152 · 0,684 · 29,293 min | **105/152 · 0,691 · 29,293 min** | 104/152 · 0,684 · 29,100 min |
| 40 min | 124/152 · 0,816 · 38,773 min | **125/152 · 0,822 · 39,040 min** | 121/152 · 0,796 · 38,267 min |
| 45 min | 126/152 · 0,829 · 43,620 min | **127/152 · 0,836 · 43,920 min** | 123/152 · 0,809 · 43,050 min |
| 50 min | 127/152 · 0,836 · 47,802 min | **129/152 · 0,849 · 46,959 min** | 125/152 · 0,822 · 47,402 min |
| 58 min | 129/152 · 0,849 · 52,800 min | **132/152 · 0,868 · 52,182 min** | 127/152 · 0,836 · 52,812 min |

`NO_RESCUE` apresentou a maior cobertura pontual nos cinco orçamentos nesta população. Isso descreve as políticas Nested avaliadas em TRAIN; não é uma estimativa do resultado de cada configuração fixa nem prova de superioridade universal.

## 5. Bootstrap e fronteira Pareto

O bootstrap usou 10.000 reamostragens pareadas por `match_id`, seed `901006`, aplicando a mesma amostra de partidas aos procedimentos comparados. A fronteira contém somente os 15 pontos Nested das três famílias e cinco orçamentos. Os intervalos descrevem a estabilidade entre partidas TRAIN; não constituem teste formal de superioridade e não abrangem toda a incerteza de treinamento e seleção.

Para `CORE_RESCUE − NO_RESCUE`, as diferenças de recall foram −0,7, −2,6, −2,6, −2,6 e −3,3 pontos percentuais nos orçamentos 30, 40, 45, 50 e 58 minutos. Os respectivos intervalos percentis de 95% foram [−6,2; 5,8], [−5,9; 0,0], [−5,1; −0,6], [−5,0; −0,6] e [−6,8; −0,6] pontos percentuais.

## 6. Configurações fixas V2

As configurações foram escolhidas no grid congelado usando o TRAIN completo. Cobertura e duração desta tabela são resultados **in-sample de seleção**, não desempenho esperado em dados independentes.

| Modo | Família | Método | Janela | K | Cobertura in-sample | Minutos médios / máximo |
|---|---|---|---:|---:|---:|---:|
| FAST_V2 | NO_RESCUE | TEMPORAL | 45 s | 40 | 140/152 | 30,000 / 30,000 |
| BALANCED_V2 | CORE_RESCUE | CORE40_RRF4_NET_RESCUE | 60 s | 40 | 144/152 | 39,308 / 39,853 |
| HIGH_RECALL_V2 | CORE_RESCUE | CORE40_RRF4_NET_RESCUE | 60 s | 40 | 144/152 | 39,308 / 39,853 |

BALANCED_V2 e HIGH_RECALL_V2 selecionaram a mesma configuração concreta; portanto, não há diferença de cobertura observada entre eles no conjunto de seleção. Os limites de orçamento são tetos, não objetivos de duração.

## 7. Preflight e freeze

A primeira tentativa do preflight parou porque o snapshot intermediário v45 esperava um hash anterior de `plano_tcc_jira.md`. Nenhum resultado do Nested foi alterado. Foi criado um novo snapshot intermediário, preservando versões anteriores e registrando os artefatos full-pool. O preflight histórico passou com 172 artefatos no v46; após a inclusão do finalizador V2 e as atualizações documentais, o v48 registrou 175 artefatos.

No v48, o preflight histórico confirmou os gates do sprint, quatro execuções de tracking e os resultados Report 90 históricos, sem promover modos ao V4 e sem abrir TEST. O preflight específico V2 confirmou dimensões Nested, bootstrap, Pareto, configurações e hashes dos quatro modelos. O manifesto final V2 foi escrito em `experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v2.json`, com fonte no v48 (SHA-256 `5fd5a65c2cfd2800d9180785817f754c71b0b9b0050cbd81157611389f0597bb`). TEST está declarado fechado até o freeze; V4 permaneceu inalterado e não promovido.

**Nota de rastreabilidade:** no momento da execução, o diretório canônico não continha `final_freeze_manifest_v1.json`; havia um preflight aprovado e snapshots intermediários. O campo `supersedes` do manifesto V2 registra o nome esperado para V1, mas o estado anterior verificável era o snapshot intermediário/preflight, não um manifesto final V1 presente no diretório.

## 8. Estado de TEST após o freeze

Depois de gravado o freeze V2, foi consultado o manifesto de partição e inventariado o diretório de vídeos TEST: são nove partidas e 57 gols na contagem de catálogo. Não foi encontrado catálogo timestampado humano de TEST. Apenas TRAIN e VALIDATION tinham manifests anotados no checkout. Nenhum score, ranking, shortlist ou resultado dos modos V2 foi gerado em TEST.

Foi criado o template `experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V1/goal_catalog_timestamp_manifest_v1.csv`, com 57 linhas pendentes, e o anotador `experiments/ball/annotate_test_goal_catalog_timestamps_v1.py`, que exige o freeze V2 e só aceita o split `test`. A avaliação final continua pendente da revisão humana dos timestamps e da geração do CSV anotado. `not_evaluable` e divergências de catálogo devem ser registrados, não tratados como ausência de gol.

Quando o catálogo for anotado, a avaliação única deverá aplicar os três modos congelados sem retuning, reportar recall apenas sobre gols concluídos, minutos únicos, segmentos, candidatos, tempo de processamento e ganhos/perdas por partida. Não se deve escolher um vencedor após observar TEST.

## 9. Arquivos de referência e reprodução

- Plano/protocolo: `docs/plans/94_plano_execucao_nested_lomo_fullpool_25_train_14-10-2026.md`.
- Resultados Nested, bootstrap e Pareto: `docs/plans/95_resultados_nested_lomo_fullpool_25_train_14-10-2026.md`.
- Preparação de anotação TEST: `docs/plans/96_estado_pos_freeze_e_prontidao_test_14-10-2026.md`.
- Dados recuperados e montados: `results/net_motion_features_train_fullpool_recovered_v1/`, `results/rich_temporal_dataset_train_fullpool_nested_v1/` e `results/net_motion_dataset_train_fullpool_nested_v1/`.
- Nested: `experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1/`.
- Bootstrap/Pareto: `experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1/`.
- Modos/modelos V2: `experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/`.
- Scripts de montagem, Nested, consolidação, seleção e freeze: `assemble_net_motion_dataset_fullpool_v1.py`, `evaluate_nested_time_budget_lomo_fullpool_v1.py`, `summarize_nested_fullpool_oof_v1.py`, `select_nested_fullpool_modes_v2.py` e `finalize_report94_fullpool_v2.py`, todos em `experiments/ball/`.

Não foi executada a suíte automatizada do aplicativo nesta atividade; o trabalho realizado foi a avaliação experimental e a preparação metodológica do TEST.
