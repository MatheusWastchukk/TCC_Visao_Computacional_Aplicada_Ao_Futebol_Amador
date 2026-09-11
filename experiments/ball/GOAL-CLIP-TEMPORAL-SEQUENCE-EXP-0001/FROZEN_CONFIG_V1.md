# Configuração congelada V1

## Decisão

A sequência temporal é promovida sobre o baseline agregado apenas como **verificador/rankeador experimental**. Ela não está aprovada como filtro binário operacional.

## Configuração

- Dataset de desenvolvimento: `GOAL-CLIP-VERIFIER-DATASET-V2/audit_v1/resolved_v1/goal_clip_verifier_dataset_v2_audited.csv`, somente `TRAIN`.
- Encoder: ResNet18 ImageNet congelada.
- Entrada: oito embeddings ordenados de 512 dimensões, sem agregação temporal prévia (4.096 atributos).
- Classificador: regressão logística L2 com `class_weight="balanced"`, `max_iter=3000` e `random_state=42`.
- Protocolo: leave-one-match-out em 25 partidas.

## Evidência de promoção

| Configuração | AP | AUC ROC |
| --- | ---: | ---: |
| Baseline agregado, floresta aleatória | 66,68% | 64,94% |
| Sequência ordenada, regressão logística | 72,03% | 74,90% |

A AP aumentou 5,35 pontos percentuais, acima da regra congelada de 3 pontos. Para recall de 84,87%, a taxa de falsos positivos caiu de 68,42% para 48,03%.

## Limite operacional

O limiar de desenvolvimento 0,06779222 atinge 84,87% de recall, mas ainda produz 48,03% de taxa de falsos positivos. Sob máximo de 20% de falsos positivos, o recall é 50,00% no limiar 0,85399765. Portanto, não integrar como decisão binária automática nem ajustar mais parâmetros em `TRAIN`.

Uma confirmação posterior deve reutilizar exatamente esta arquitetura congelada em um protocolo separado; ela não autoriza abertura de `TEST`.
