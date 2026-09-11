# GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001

Experimento preparado, ainda não executado.

## Hipótese congelada

A ordem temporal dos oito frames uniformes pode distinguir um gol de um não-gol melhor que a agregação por média, desvio-padrão e diferenças do baseline pré-treinado anterior.

## Configuração

- Dataset: `GOAL-CLIP-VERIFIER-DATASET-V2/audit_v1/resolved_v1/goal_clip_verifier_dataset_v2_audited.csv`.
- Escopo: somente `TRAIN`, com leave-one-match-out.
- Encoder: ResNet18 ImageNet congelada; os pesos já estão no cache local.
- Sequência: oito embeddings de 512 dimensões preservados em ordem, totalizando 4.096 atributos por clipe.
- Classificador: regressão logística L2 com balanceamento de classes.

## Regra de promoção

Somente considerar a sequência promissora se superar a AP de 66,68% do baseline agregado em pelo menos 3 pontos percentuais e melhorar o compromisso entre recall e falsos positivos. Caso contrário, registrar NO-GO e não abrir `VALIDATION`.

## Comando preparado

```powershell
& .\.venv\Scripts\python.exe .\experiments\ball\evaluate_goal_clip_temporal_sequence_lomo_v1.py `
  --dataset .\experiments\ball\GOAL-CLIP-VERIFIER-DATASET-V2\audit_v1\resolved_v1\goal_clip_verifier_dataset_v2_audited.csv `
  --videos-dir .\data\Videos_Edited `
  --output .\experiments\ball\GOAL-CLIP-TEMPORAL-SEQUENCE-EXP-0001\run_v1
```
