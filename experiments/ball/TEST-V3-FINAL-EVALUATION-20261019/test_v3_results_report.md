# Resultados da avaliação final V3 em TEST

**Data de fechamento:** 19/10/2026 (horário de São Paulo)<br>
**Estado:** avaliação única concluída; políticas mantidas congeladas.<br>
**População:** nove partidas TEST, 57 timestamps humanos concluídos.<br>
**Pergunta:** qual fração dos eventos catalogados está dentro do vídeo entregue pelas políticas congeladas?<br>
**Atenção:** isto mede cobertura dos gols catalogados. Não mede precisão de detecção automática nem eficácia clínica do sistema.

## Integridade e protocolo

- Catálogo cego: `GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/goal_catalog_timestamp_manifest_frozen_v1.csv`; 57/57 registros concluídos, nove partidas; SHA-256 `593fabd777d0e0cf99f723d7a3853d7c8c5d08e951ce54224300204fd824f1b4`.
- O operador confirmou que anotou os vídeos originais e o catálogo, sem consultar saídas da IA, e marcou o instante visual do gol. Em `match_40`, o gol 006 foi revisto porque passou despercebido na primeira passagem; o timestamp revisto foi confirmado no vídeo e congelado antes da inferência.
- Protocolo pré-registrado: `GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/test_evaluation_protocol_v1.md`.
- Inferência: `attempt_02/inference_manifest_test_v1.json`; nove partidas, 1.757 grupos temporais pontuados. O catálogo humano não foi carregado pelo processo de inferência (`catalog_loaded_by_inference=false`).
- Scores: `attempt_02/ranked_candidates_test_v1.csv`; seleções: `attempt_02/selected_candidates_test_v1.csv`; saídas por partida: `attempt_02/per_match_inference_test_v1.csv`.
- Métricas: `metrics_v1/summary_test_v1.json`, `metrics_v1/goal_coverage_test_v1.csv`, `metrics_v1/per_match_test_v1.csv` e `metrics_v1/paired_mode_gains_test_v1.csv`.
- Os artefatos e os quatro modelos correspondiam aos hashes congelados. Dispositivo: AMD Radeon RX 9060 XT, ambiente PyTorch `2.9.1+rocm7.2.1` / TorchVision `0.24.1+rocm7.2.1`.

## Resultado agregado

| Política congelada | Gols cobertos | Cobertura | Minutos únicos por partida (média) | Vídeo original revisado (agregado) | Segmentos unidos por partida (média) | Partidas acima do teto |
|---|---:|---:|---:|---:|---:|---:|
| FAST_V3 (30 min) | 45/57 | 78,95% | 27,27 | 34,68% | 54,78 | 0/9 |
| BALANCED_V3 (40 min) | 50/57 | 87,72% | 36,34 | 46,21% | 54,78 | 0/9 |
| HIGH_RECALL_V3 (58 min) | 54/57 | 94,74% | 49,60 | 63,07% | 38,22 | 0/9 |

Todos os jogos ficaram dentro dos tetos de 30, 40 e 58 minutos conforme o manifesto congelado. A comparação dos minutos únicos foi feita após unir intervalos e limitar os trechos à duração real de cada vídeo. Os candidatos foram agrupados quando seus intervalos se sobrepunham ou se encostavam.

## Resultados por partida

Cada célula mostra `gols cobertos / gols catalogados (minutos únicos entregues)`.

| Partida | FAST_V3 | BALANCED_V3 | HIGH_RECALL_V3 |
|---|---:|---:|---:|
| match_01 | 5/7 (29,67 min) | 6/7 (39,52 min) | 7/7 (51,76 min) |
| match_11 | 3/3 (15,67 min) | 3/3 (20,89 min) | 3/3 (29,83 min) |
| match_17 | 1/3 (30,00 min) | 3/3 (40,00 min) | 3/3 (54,22 min) |
| match_19 | 10/14 (24,39 min) | 11/14 (32,44 min) | 13/14 (46,46 min) |
| match_29 | 8/9 (29,76 min) | 9/9 (39,65 min) | 9/9 (53,69 min) |
| match_31 | 2/3 (29,98 min) | 2/3 (39,92 min) | 3/3 (53,71 min) |
| match_34 | 5/6 (26,00 min) | 5/6 (34,67 min) | 6/6 (50,26 min) |
| match_39 | 6/6 (30,00 min) | 6/6 (40,00 min) | 5/6 (53,50 min) |
| match_40 | 5/6 (30,00 min) | 5/6 (40,00 min) | 5/6 (52,99 min) |

## Diferenças pareadas

- BALANCED_V3 contra FAST_V3: 45 gols cobertos por ambos, 5 somente por BALANCED_V3, nenhum somente por FAST_V3 e 7 por nenhum dos dois.
- HIGH_RECALL_V3 contra FAST_V3: 44 por ambos, 10 somente por HIGH_RECALL_V3, 1 somente por FAST_V3 e 2 por nenhum.
- HIGH_RECALL_V3 contra BALANCED_V3: 49 por ambos, 5 somente por HIGH_RECALL_V3, 1 somente por BALANCED_V3 e 2 por nenhum.

HIGH_RECALL_V3 cobriu mais gols no agregado, mas entregou mais vídeo. Há eventos cobertos por FAST_V3 ou BALANCED_V3 que HIGH_RECALL_V3 não cobriu, então as coberturas não são estritamente aninhadas. Não declarar um modo vencedor nem alterar os modos com base neste TEST.

## Custo computacional observado

| Etapa | Tempo registrado |
|---|---:|
| Mineração dos nove vídeos | 10.846,13 s (aprox. 3 h 00 min 46 s) |
| Features temporais | 1,52 s |
| Movimento local da rede | 466,03 s (aprox. 7 min 46 s) |
| Embeddings visuais | 1.826,54 s (aprox. 30 min 27 s) |
| Pontuação dos modelos | 0,34 s |
| Políticas e união de intervalos | 0,08 s |

O tempo de mineração é a soma dos cronômetros por partida gravados pelo minerador. O primeiro início da tentativa 02 precisou redirecionar os caches MIOpen para diretórios graváveis do projeto após uma falha de permissão fora da área de trabalho. Uma incompatibilidade entre a chamada Python e a interface CLI do extrator foi corrigida depois da mineração; a continuação reutilizou a mineração completa e não repetiu inferência. Nenhum parâmetro de modelo ou modo foi alterado.

## Correções e ressalvas de auditoria

1. A primeira execução de métricas interpretou a string CSV `"False"` como verdadeiro em Python ao calcular `matches_over_budget`. O conversor de booleanos foi corrigido e as métricas determinísticas foram recalculadas a partir dos mesmos arquivos de inferência. O resultado correto é zero partidas acima do teto em todos os modos.
2. O resumo gerado pelo extrator de movimento preservou o texto histórico de escopo “diagnóstico TRAIN”, embora tenha recebido e processado o CSV de frames TEST. Corrigi o campo textual do `net_motion_features_test_v1/metrics.json` e registrei a correção; nenhuma linha de feature mudou. A etapa de inferência e o manifesto confirmam que os nove arquivos e os 1.757 grupos avaliados são TEST. O texto genérico do helper deve ser parametrizado antes de reutilizá-lo em outro experimento.
3. Uma tentativa anterior falhou na inicialização do MIOpen antes de processar quadros; está registrada separadamente em `attempt_01`. Não produziu scores. A tentativa válida foi `attempt_02` e ficou auditada no manifesto de ambiente.

## Interpretação e próximos passos

Os resultados mostram a cobertura observada no TEST para as três políticas fixas. Eles não medem precisão, tempo de revisão humana ou usabilidade em estudo controlado. Em particular, o fato de HIGH_RECALL_V3 cobrir 54/57 não transforma o sistema em detector definitivo de gols; continua sendo uma ferramenta de mineração/priorização para revisão humana.

Manter os três modos congelados e relatar inclusive as perdas específicas de cada um. Atualizar o artigo em seção separada dos resultados Nested OOF em TRAIN; discutir a exposição histórica de duas partidas TEST conforme os relatórios anteriores. Não usar estes resultados para retreinar, retunar, escolher um vencedor ou alterar o ranking.
