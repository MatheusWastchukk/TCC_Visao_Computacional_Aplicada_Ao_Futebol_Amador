# Relatório 95 — Nested LOMO full-pool em 25 partidas TRAIN

**Data:** 14/10/2026<br>
**Plano executado:** [94 — Nested LOMO full-pool em 25 partidas TRAIN](94_plano_execucao_nested_lomo_fullpool_25_train_14-10-2026.md)<br>
**Estado:** análise e seleção TRAIN concluídas; preflight de freeze V2 ainda pendente<br>
**Limite:** TEST e VALIDATION não foram acessados.

## Resumo

A execução ampliou o Nested LOMO da coorte anterior de 22 partidas/130 gols avaliáveis para 25 partidas, 5.742 grupos candidatos e 152 timestamps concluídos. A matriz `net_motion` recebeu apenas os 370 grupos ausentes; os 5.372 registros históricos e os resultados antigos foram preservados. O Nested produziu estimativas outer-fold em TRAIN para cinco orçamentos, três famílias e um grid de políticas já congelado. Em seguida foram calculados bootstrap pareado por partida e Pareto, e selecionadas configurações operacionais fixas V2 usando o mesmo grid.

O primeiro preflight do freeze canônico parou por divergência de hash: o snapshot intermediário v45 espera uma versão anterior de `plano_tcc_jira.md`. Isso não invalida os cálculos, mas impede declarar o freeze até gerar e verificar um novo snapshot. TEST permanece fechado enquanto isso não for resolvido.

## Coorte e integridade dos dados

- Catálogo: 152 gols avaliáveis (`annotation_status=completed`) em 25 partidas TRAIN.
- População comum: 5.742 grupos, com pareamento um-para-um entre features temporais, `net_motion`, scores visuais e labels.
- Recuperação: 370 grupos de `match_02`, `match_04` e `match_06`, gerados com o extrator e agrupamento já existentes (`frame_gap_seconds=0.2`).
- Comparação histórica: nos 5.372 grupos já presentes na matriz `net_motion`, não houve diferença nas métricas, features temporais derivadas, labels ou timestamps. Não houve chaves extras ou ausentes.
- Escopo de dados: somente TRAIN. A reconstrução e os resultados novos foram escritos em diretórios separados; não se sobrescreveu a matriz antiga nem os Nested de 22 partidas.

## Resultado Nested por orçamento

Cada célula mostra gols cobertos dentre 152 (recall) e minutos únicos médios de vídeo por partida. A regra de orçamento foi atendida em todos os 25 jogos retidos em cada condição.

| Orçamento | GLOBAL_BEST | NO_RESCUE | CORE_RESCUE |
|---:|---:|---:|---:|
| 30 min | 104/152 (0,684); 29,293 min | **105/152 (0,691); 29,293 min** | 104/152 (0,684); 29,100 min |
| 40 min | 124/152 (0,816); 38,773 min | **125/152 (0,822); 39,040 min** | 121/152 (0,796); 38,267 min |
| 45 min | 126/152 (0,829); 43,620 min | **127/152 (0,836); 43,920 min** | 123/152 (0,809); 43,050 min |
| 50 min | 127/152 (0,836); 47,802 min | **129/152 (0,849); 46,959 min** | 125/152 (0,822); 47,402 min |
| 58 min | 129/152 (0,849); 52,800 min | **132/152 (0,868); 52,182 min** | 127/152 (0,836); 52,812 min |

`NO_RESCUE` teve a maior cobertura pontual nos cinco orçamentos. Isso é uma comparação de procedimentos de seleção Nested em TRAIN; não demonstra que cada componente de resgate seja inútil em outros critérios, populações ou aplicações. A fronteira Pareto inclui somente os 15 pontos acima, sem misturar resultados exploratórios de outros protocolos.

## Incerteza pareada e alcance da evidência

Foram feitas 10.000 reamostragens de partidas, seed `901006`, usando as mesmas partidas em cada par de políticas. O bootstrap é descritivo e não é um teste formal de superioridade. Para `CORE_RESCUE − NO_RESCUE`, a diferença pontual de recall foi −0,7, −2,6, −2,6, −2,6 e −3,3 pontos percentuais nos orçamentos 30, 40, 45, 50 e 58 min. Os intervalos percentis de 95% correspondentes foram:

| Orçamento | Diferença de recall (pontos percentuais) | IC bootstrap de 95% |
|---:|---:|---:|
| 30 min | −0,7 | [−6,2; 5,8] |
| 40 min | −2,6 | [−5,9; 0,0] |
| 45 min | −2,6 | [−5,1; −0,6] |
| 50 min | −2,6 | [−5,0; −0,6] |
| 58 min | −3,3 | [−6,8; −0,6] |

Esses intervalos capturam variação entre as partidas reamostradas para os resultados outer-fold já obtidos. Não incluem toda a incerteza de treino e seleção e não devem ser usados como estimativas independentes de VALIDATION ou TEST.

## Modos operacionais fixos V2

As configurações foram selecionadas em todo TRAIN dentro dos valores do grid pré-existente. O critério escolhe uma configuração concreta para operação; seus números de cobertura são **in-sample** e não são estimativas de generalização.

| Modo | Orçamento | Família de seleção | Método / janela / K | Cobertura in-sample | Minutos médios / máximo |
|---|---:|---|---|---:|---:|
| FAST_V2 | 30 min | NO_RESCUE | TEMPORAL / 45 s / 40 | 140/152 (0,921) | 30,000 / 30,000 |
| BALANCED_V2 | 40 min | CORE_RESCUE | CORE40_RRF4_NET_RESCUE / 60 s / 40 | 144/152 (0,947) | 39,308 / 39,853 |
| HIGH_RECALL_V2 | 58 min | CORE_RESCUE | CORE40_RRF4_NET_RESCUE / 60 s / 40 | 144/152 (0,947) | 39,308 / 39,853 |

BALANCED_V2 e HIGH_RECALL_V2 selecionaram a mesma configuração e, portanto, entregam a mesma cobertura observada no conjunto usado para seleção; não se deve inferir que HIGH_RECALL V2 ofereça ganho sobre BALANCED V2. Os limites de 40 e 58 minutos são tetos, não metas obrigatórias de duração.

## Artefatos reproduzíveis

- Protocolo e critérios: `docs/plans/94_plano_execucao_nested_lomo_fullpool_25_train_14-10-2026.md`.
- Dataset `net_motion` reconstruído: `results/net_motion_dataset_train_fullpool_nested_v1/temporal_group_dataset_net_motion_v1.csv`.
- Nested e manifesto de entradas: `experiments/ball/NESTED-TIME-BUDGET-FULLPOOL-25-TRAIN-V1/`.
- Sumarização, bootstrap e Pareto: `experiments/ball/REPORT94-FULLPOOL-CONSOLIDATION-V1/`.
- Modos, modelos e manifests: `experiments/ball/REPORT94-FULLPOOL-FINAL-MODES-V2/`.
- Scripts versionados: `assemble_net_motion_dataset_fullpool_v1.py`, `evaluate_nested_time_budget_lomo_fullpool_v1.py`, `summarize_nested_fullpool_oof_v1.py` e `select_nested_fullpool_modes_v2.py`, em `experiments/ball/`.

## Decisão e próximos gates

1. Manter os modos V2 como candidatos congelados, separados dos modos V1 históricos; não escolher novamente com base em TEST.
2. Atualizar o snapshot intermediário com hashes de todos os artefatos deste resultado e executar novamente o preflight canônico, que também verifica as evidências anteriores do sprint.
3. Se todo o preflight passar, gravar um manifesto final V2 antes de abrir TEST. Então executar uma única avaliação com parâmetros registrados, sem selecionar vencedor nem retunar após observar TEST.
4. Se o preflight ou qualquer validação de integridade falhar, manter TEST fechado e corrigir a rastreabilidade antes de prosseguir.
