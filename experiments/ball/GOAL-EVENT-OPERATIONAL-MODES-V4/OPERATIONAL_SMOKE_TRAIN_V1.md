# Demonstração operacional TRAIN V1

**Data:** 15/09/2026
**Configuração:** `cobertura_reduzida` V4 congelada (`F30+T15`, janelas 20+40)
**Escopo:** duas partidas `TRAIN`; nenhuma etapa desta demonstração abriu `VALIDATION` ou `TEST`.

## Objetivo

Demonstrar o fluxo operacional completo, sem retreinar ou recalibrar modelos:

```text
vídeo TRAIN
→ mineração espacial
→ agrupamento e ranking temporal
→ verificador visual e fusão congelados
→ segmentos POSSIBLE_GOAL
→ revisão humana
→ associação aos timestamps TRAIN
```

Os resultados não são uma nova estimativa de generalização. As partidas foram uma amostra pequena de demonstração operacional em `TRAIN` e não podem substituir as métricas LOMO em `TRAIN`, a confirmação previamente concluída em `VALIDATION` ou a avaliação final reservada de `TEST`.

## Resultados revisados

| Partida | Gols válidos catalogados | Candidatos | Segmentos | Minutos únicos | GOAL / NO_GOAL / NOT_EVALUABLE | Gols cobertos | Gols confirmados |
|---|---:|---:|---:|---:|---:|---:|---:|
| `match_04` | 7 | 43 | 36 | 42,11 | 7 / 29 / 0 | 7/7 | 7/7 |
| `match_02` | 9 | 45 | 39 | 44,12 | 9 / 30 / 0 | 9/9 | 9/9 |
| **Agregado** | **16** | **88** | **75** | **86,23** | **16 / 59 / 0** | **16/16** | **16/16** |

Em `match_04`, um oitavo item histórico do catálogo está registrado como `catalog_discrepancy` e, portanto, não entra no denominador de sete gols válidos.

## Artefatos locais de origem

Os clipes e CSVs brutos permanecem locais, pois são artefatos gerados e incluem vídeo. A síntese leve versionada está em `operational_smoke_train_summary_v1.csv`.

- `results/smoke_match_04_v4/results.json`
- `results/smoke_match_04_v4/clips/clip_annotations.csv`
- `results/smoke_match_04_v4/review_evaluation.json`
- `results/operational_match_02_v4/results.json`
- `results/operational_match_02_v4/clips/clip_annotations.csv`
- `results/operational_match_02_v4/review_evaluation.json`

## Reprodução

Para executar o pipeline em uma partida `TRAIN`, use `app/analyze_match.py`. A entrada valida o manifesto e rejeita qualquer vídeo fora de `TRAIN`. Depois, use `app/generate_segment_clips.py`, `app/annotate_segment_clips.py` e `app/evaluate_reviewed_segments.py` nesta ordem.

Os pesos, os modos e seus resultados de desenvolvimento/validação permanecem definidos em `FROZEN_CONFIG_V4.md`. Não ajustar parâmetros a partir destes dois smokes antes da avaliação final planejada.
