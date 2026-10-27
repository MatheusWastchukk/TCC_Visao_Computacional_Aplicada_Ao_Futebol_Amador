# Relatório 96 — Freeze V2 e prontidão da avaliação TEST

**Data:** 14/10/2026<br>
**Freeze:** [manifesto final V2](../../experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v2.json)<br>
**Relatórios do Nested:** [Plano 94](94_plano_execucao_nested_lomo_fullpool_25_train_14-10-2026.md) e [Resultados 95](95_resultados_nested_lomo_fullpool_25_train_14-10-2026.md).

## Estado

O Nested LOMO full-pool em TRAIN, o bootstrap, a curva Pareto e a seleção dos três modos fixos V2 foram concluídos. O preflight legado foi aprovado e o preflight específico V2 confirmou as dimensões 25 partidas/5.742 grupos/152 gols, 25 folds, ausência de violações de orçamento, hashes dos modelos e as configurações FAST/BALANCED/HIGH_RECALL. O manifesto final V2 foi gravado na mesma raiz canônica do freeze anterior e declara TEST fechado até o freeze, V4 inalterado e não promovido.

Depois do freeze, a inspeção do split TEST encontrou nove vídeos, com 57 gols registrados no manifesto de partição. O repositório **não contém timestamps humanos de TEST**: os catálogos timestampados existentes são de TRAIN e VALIDATION. Sem timestamps independentes, não é possível calcular recall de gols nem ganhos/perdas por partida. Por isso, os modelos V2 ainda não foram executados sobre TEST. O pipeline não deve gerar resultados antes de a anotação independente estar pronta, para que a revisão dos timestamps não seja influenciada pelas predições.

## Template e anotação

Foi preparado `experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V1/goal_catalog_timestamp_manifest_v1.csv` com uma linha pendente por gol catalogado (57 linhas em 9 partidas). O template não contém timestamps nem resultados do sistema. O anotador específico verifica que o manifesto de freeze V2 existe e aceita somente o split `test`.

Com OpenCV disponível em ambiente gráfico, a revisão pode ser feita a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe experiments\ball\annotate_test_goal_catalog_timestamps_v1.py `
  --input experiments\ball\GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V1\goal_catalog_timestamp_manifest_v1.csv `
  --output experiments\ball\GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V1\goal_catalog_timestamp_manifest_v1_annotated.csv `
  --videos-dir data\Videos_Test_Fechado
```

No anotador: `A/D` avança/retrocede 5 s, `J/L` avança/retrocede 30 s, `G` registra o gol, `U` marca não avaliável, `X` registra divergência de catálogo, `W` salva e `Q` sai. Os estados `not_evaluable` e `catalog_discrepancy` devem ser revisados e excluídos do denominador conforme o protocolo; não presumir que os 57 registros sejam todos avaliáveis.

Após concluir e validar o catálogo, deve-se executar uma única avaliação cega aos resultados, com os modos e modelos já congelados. Reportar recall do catálogo concluído, minutos únicos de vídeo, segmentos, candidatos, tempo de processamento e ganhos/perdas por partida; não retunar nem selecionar vencedor com base em TEST.

## Artefatos

- `experiments/ball/REPORT87-PD-SPRINT-INTERIM-FREEZE-20261005/interim_freeze_manifest_v48.json`: snapshot fonte validado pelo preflight.
- `experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v2.json`: freeze TRAIN final e protocolo dos modos V2.
- `experiments/ball/prepare_test_goal_catalog_manifest_v1.py`: gerador do template pendente a partir do split manifest.
- `experiments/ball/annotate_test_goal_catalog_timestamps_v1.py`: anotador que só aceita TEST após o freeze V2.
