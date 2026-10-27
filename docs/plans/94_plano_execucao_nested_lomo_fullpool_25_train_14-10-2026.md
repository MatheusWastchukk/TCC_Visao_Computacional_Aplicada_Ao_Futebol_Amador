# Plano 94 — Nested LOMO full-pool em 25 partidas TRAIN

**Data:** 14/10/2026<br>
**Estado:** Nested, bootstrap, Pareto e seleção fixa V2 concluídos; preflight de freeze V2 pendente<br>
**Escopo:** ampliar a análise Nested de 22 partidas/130 timestamps para a população TRAIN avaliável de 25 partidas/152 timestamps, sem alterar o espaço de políticas.<br>
**TEST:** fechado. Nenhum arquivo, rótulo, vídeo ou anotação de TEST será lido nesta execução.

## 1. Motivação

Os resultados Nested existentes usam uma coorte comum de 22 partidas porque a matriz `net_motion` então disponível cobria somente esses jogos. As matrizes TRAIN full-pool e os timestamps avaliáveis passaram a cobrir 25 partidas. Este plano reexecuta o Nested antes do freeze final e antes de TEST para que a análise principal use a população TRAIN originalmente definida.

Os resultados de 22 partidas permanecem preservados como evidência histórica da coorte anterior; os arquivos não serão sobrescritos. A extensão 25/25 só será considerada válida se o novo pareamento atingir 5.742/5.742 `group_uid`, com 25 partidas e 152 timestamps avaliáveis.

## 2. Pré-auditoria observada

| Entrada | Estado observado |
|---|---|
| Features temporais full-pool | 5.742 grupos em 25 partidas. |
| Scores visuais e embeddings de candidatos | Scores para 5.742 grupos em 25 partidas; manifesto e tensor serão checados antes do Nested. |
| Predições/embeddings do verificador visual | Predições TRAIN em 25 partidas; quantidade, classes e alinhamento serão verificados antes do Nested. |
| Catálogo TRAIN full-pool | 152 timestamps concluídos em 25 partidas. |
| Matriz `net_motion` histórica | 5.372 grupos em 22 partidas. |
| Dados de quadros recuperados | `candidate_frames_v1.csv` do piloto contém `match_02`, `match_04` e `match_06`; ao agrupar com a regra existente, forma 370 grupos. |
| TEST | A documentação canônica ainda registra TEST fechado e freeze final não emitido. |

## 3. Reconstrução de entradas

1. Manter intactos os 5.372 registros `net_motion` históricos.
2. Regenerar somente os 370 grupos de `match_02`, `match_04` e `match_06` com `extract_net_motion_features_v1.py`, o mesmo agrupamento temporal existente e `frame_gap_seconds=0.2`.
3. Derivar as 13 features de trajetória usando `prepare_goal_event_rich_temporal_dataset_v1.py` sobre os quatro bins full-pool existentes, sem inferência nova.
4. Montar um dataset full-pool novo, combinando o dataset rico de 5.742 grupos com os 5.372 registros históricos e os 370 registros reconstruídos.
5. Exigir unicidade, igualdade de rótulos fracos, pareamento um-para-um e ausência de chaves extras ou faltantes. Para os 5.372 grupos históricos, exigir que as features base derivadas coincidam com a fonte histórica antes de aceitar a montagem.

As novas matrizes serão gravadas em diretórios próprios. `results/net_motion_dataset_train_v1/` e os resultados Nested antigos não serão alterados.

## 4. Protocolo Nested congelado

Usar a implementação de `evaluate_nested_time_budget_lomo_v1.py` sem modificar modelos, fórmulas, critérios de seleção ou orçamento. A nova execução será registrada em diretório separado e aceitará somente TRAIN.

| Parâmetro | Valores congelados |
|---|---|
| Outer folds | 25, uma partida retida por fold. |
| Grupos | 5.742 após pareamento completo. |
| Timestamps avaliáveis | 152, todos com `annotation_status=completed`. |
| Métodos base | `TEMPORAL`, `RRF3`, `RRF4_NET`, `CORE35_FUSION10_NET10`, `CORE40_RRF4_NET_RESCUE`. |
| K por partida | 20, 30, 40, 50, 55 e 60. |
| Janelas | 30, 40, 45 e 60 segundos. |
| Separação entre candidatos | gap de 45 segundos. |
| Orçamentos | 30, 40, 45, 50 e 58 minutos por partida. |
| Modelos | RF com 300 árvores, `min_samples_leaf=2`, `class_weight=balanced`, `random_state=42`; regressões logísticas e embeddings congelados conforme o script histórico. |
| Seleção interna | Política precisa respeitar o limite em todas as partidas internas; maximizar gols cobertos, depois minimizar minutos únicos médios, segmentos e complexidade fixa. |
| Comparação | `GLOBAL_BEST`, `NO_RESCUE` e `CORE_RESCUE` selecionados separadamente no mesmo outer fold/orçamento. |

Em cada outer fold, a partida retida não participa de treino de rankers, verificador/fusão, normalização ou escolha de política. Scores base serão gerados dentro do fold, seguindo a implementação Nested rigorosa existente. Os resultados de 22 partidas não serão reutilizados como scores do novo outer fold.

## 5. Consolidação e decisões posteriores

Após os 25 folds:

1. Gerar cobertura OOF por timestamp e por partida para as três famílias e os cinco orçamentos.
2. Recalcular bootstrap pareado por partida com 10.000 reamostragens e seed `901006`; usar a mesma amostra de partidas para todas as políticas comparadas.
3. Recalcular a fronteira Pareto somente com pontos Nested compatíveis.
4. Reexecutar a seleção fixa dos três modos usando TRAIN e o grid já congelado. Resultados completos usados para seleção serão identificados como in-sample, separados das estimativas OOF.
5. Atualizar o snapshot e o preflight canônicos como uma revisão V2 do freeze; manter V1 e a coorte 22 como históricos.
6. Só depois de todos os artefatos, hashes, critérios e modos V2 passarem no preflight será feita a abertura única de TEST, conforme protocolo, sem escolha de vencedor ou retuning após observar os resultados.

## 6. Interpretação e limites

- A avaliação é TRAIN-only Nested LOMO; não é validação independente nem estimativa universal de generalização.
- 152 é o denominador dos timestamps avaliáveis no catálogo full-pool. Não substituir por 156 registros brutos nem confundir com recall do Candidate Pool.
- Candidate Pool, ranking full-pool e Nested sob orçamento são populações/etapas diferentes. Não esperar que a cobertura Nested iguale o teto do pool.
- Minutos únicos medem conteúdo de vídeo entregue para revisão, não tempo humano.
- Este trabalho amplia a coorte e refaz estimativas dentro do grid fixado; não abre novos métodos ou valores de hiperparâmetros.

## 7. Execução observada

- A recuperação completou 370/370 grupos nos jogos `match_02`, `match_04` e `match_06`; a montagem final contém 5.742 grupos em 25 partidas, pareados integralmente com a matriz temporal e com o catálogo de 152 gols.
- A validação dos 5.372 grupos históricos não encontrou diferenças nas features temporais derivadas, métricas `net_motion`, rótulos ou timestamps. Os novos registros foram gravados em caminhos próprios; a matriz histórica não foi sobrescrita.
- O Nested concluiu 25/25 folds: 375 linhas de partida×orçamento×família e 2.280 linhas partida×gol×orçamento×família. Cada combinação cobre os 152 gols do catálogo e não há violações do orçamento nos jogos retidos.
- A família `NO_RESCUE` cobriu 105/152, 125/152, 127/152, 129/152 e 132/152 gols para 30, 40, 45, 50 e 58 minutos, respectivamente. São estimativas do procedimento Nested LOMO em TRAIN.
- O bootstrap pareado por partida foi calculado com 10.000 reamostragens e seed `901006`; a fronteira Pareto contém os 15 pontos Nested compatíveis. Os intervalos são descritivos de estabilidade entre partidas, não provas formais de superioridade.
- A seleção no grid congelado produziu FAST V2=`TEMPORAL/45s/K40`, BALANCED V2=`CORE40_RRF4_NET_RESCUE/60s/K40` e HIGH_RECALL V2=`CORE40_RRF4_NET_RESCUE/60s/K40`. As métricas in-sample de seleção foram 140/152 e 144/152 para os dois últimos, e não são estimativas de generalização. BALANCED e HIGH_RECALL resultaram na mesma configuração concreta.
- Os resultados e limitações estão registrados em `docs/plans/95_resultados_nested_lomo_fullpool_25_train_14-10-2026.md`. A primeira tentativa de preflight detectou que o snapshot intermediário v45 tem um hash antigo para `plano_tcc_jira.md`; nenhum dado de TEST foi consultado.

## 8. Critérios de conclusão

- [x] `net_motion` recuperado para os três jogos faltantes com 370/370 grupos válidos.
- [x] Dataset preparado com 5.742/5.742 grupos, 25/25 partidas e features sem diferenças históricas nos 5.372 grupos compartilhados.
- [x] 25 outer folds concluídos e 152 timestamps representados uma vez cada em cada família/orçamento correspondente.
- [x] Bootstrap pareado e fronteira Pareto regenerados a partir do novo OOF.
- [x] Modos fixos TRAIN V2 selecionados e identificados como in-sample.
- [ ] Freeze V2 validado; antes desse marco, TEST permanece fechado.

## 9. Artefatos e comandos executados

- Recuperação: `experiments/ball/extract_net_motion_features_v1.py` com os quadros do piloto e `--frame-gap-seconds 0.2`.
- Derivação: `experiments/ball/prepare_goal_event_rich_temporal_dataset_v1.py` sobre a matriz base full-pool.
- Avaliação: cópia versionada do script de time-budget Nested, com apenas caminhos, contagens esperadas e escopo de saída ajustados para 25/5.742/152.
- Sumarização: cópia versionada do consolidador Report 90, com validações de dimensão ajustadas para 25 partidas e 152 timestamps.
