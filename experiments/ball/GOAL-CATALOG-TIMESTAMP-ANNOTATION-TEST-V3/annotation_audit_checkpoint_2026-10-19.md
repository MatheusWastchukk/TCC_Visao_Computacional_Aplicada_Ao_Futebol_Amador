# Auditoria preliminar das marcações humanas TEST

**Data:** 19/10/2026<br>
**Estado:** auditoria estrutural concluída; catálogo humano congelado; avaliação preditiva V3 concluída. Ver [relatório final](../TEST-V3-FINAL-EVALUATION-20261019/test_v3_results_report.md).

## Fonte e procedimento

- Entrada: `goal_catalog_timestamp_manifest_v1.csv`.
- Saída anotada: `goal_catalog_timestamp_manifest_completed_v1.csv`.
- O operador confirmou que marcou os eventos usando os vídeos originais e o catálogo, sem consultar rankings, clipes ou predições da IA, registrando o instante visual do gol.
- O anotador carrega os vídeos e o manifesto de catálogo; não carrega artefatos de ranking ou predição.

## Auditoria estrutural

- 57 registros de entrada e 57 registros anotados, distribuídos em nove partidas.
- 57 IDs únicos; todos pertencem ao split `test` e estão com estado `completed`.
- Os campos identificadores e metadados do catálogo-base foram preservados.
- Todos os timestamps são numéricos, não negativos e menores que a duração do vídeo correspondente.
- `event_timestamp_hhmmss` corresponde ao valor em segundos; não há IDs duplicados.
- SHA-256 do catálogo congelado `goal_catalog_timestamp_manifest_frozen_v1.csv`: `593fabd777d0e0cf99f723d7a3853d7c8c5d08e951ce54224300204fd824f1b4`.

## Revisão manual concluída

Em `match_40`, os timestamps não seguem a ordem dos índices do catálogo: a sequência temporal observada é `1, 2, 6, 3, 4, 5`. O operador explicou que o sexto gol foi percebido como ausente na primeira passagem, então voltou no vídeo para anotá-lo. O registro foi reaberto e conferido no vídeo original; a diferença de ordem foi aceita como correta, sem usar saídas da IA.

## Freeze V3 e gate de avaliação

- As quatro hashes SHA-256 dos modelos conferem com `model_manifest_v3.json`.
- O manifesto final contém três políticas V3 distintas, todas `RRF4_NET`, `gap_seconds=45`, sem rescue: `FAST_V3` (janela 30 s, K=60), `BALANCED_V3` (40 s, K=60) e `HIGH_RECALL_V3` (60 s, K=55).
- O checkout está no commit `92a080dab4fc136edc6430f5e15366d1b0f4d1b7`, com alterações locais pendentes; a execução futura deve registrar o estado exato do código utilizado.
- Protocolo registrado antes da inferência em `test_evaluation_protocol_v1.md`.
- Inferência concluída para nove partidas, com 1.757 grupos temporais pontuados e sem carregar o catálogo humano.
- Cobertura final: FAST_V3 45/57, BALANCED_V3 50/57 e HIGH_RECALL_V3 54/57. Nenhum modo excedeu seu teto por partida. Isso mede cobertura dos timestamps catalogados, não precisão automática de gols.
- Correções de integração e pós-processamento foram feitas sem repetir a inferência nem ajustar os modelos ou as políticas; detalhes e hashes estão no relatório final.
