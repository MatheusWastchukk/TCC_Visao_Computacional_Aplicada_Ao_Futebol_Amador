# Configuração congelada V4 — Modos operacionais após redução híbrida

**Data:** 13/09/2026
**Escopo de seleção:** `TRAIN`, com confirmação única em `VALIDATION`
**Estado de `TEST`:** fechado; não acessado nesta decisão.

## Decisão

O sistema permanece um minerador de `POSSIBLE_GOAL` para revisão humana. Ele não atribui `GOAL` automaticamente, não usa limiar binário do verificador visual e não substitui arbitragem.

Após a caracterização de cobertura e a única confirmação do redutor híbrido, ficam congelados três modos explícitos. Não realizar novo ajuste de pesos, limiares, janela, número de candidatos ou composição híbrida antes da avaliação final em `TEST`.

| Modo | Seleção | Janela | Resultado em VALIDATION | Tempo único médio |
|---|---|---|---:|---:|
| Alta cobertura | Random Forest temporal, até top 60 | 20+40 | 47/51 (92,16%) | 55,98 min/partida |
| Cobertura reduzida | Fusão top 30 + 15 complementos temporais não presentes | 20+40 | 44/51 (86,27%) | 43,59 min/partida |
| Triagem rápida | Fusão top 30 | 20+40 | 40/51 (78,43%) | 29,57 min/partida |

O modo de cobertura reduzida foi selecionado somente em `TRAIN` por uma regra pré-definida: entre `F30+T5/T10/T15/T20`, maximizar cobertura mantendo ao menos 20% de redução de minutos únicos contra o top 60 temporal; em empate, usar menor tempo único. Em `TRAIN`, `F30+T15` foi o selecionado. Em `VALIDATION`, ele preservou 44 dos 47 gols recuperados pelo pool temporal amplo, com economia de 22,1% no tempo único.

## Limites e interpretação

- O merge temporal reduz pouco o custo porque os candidatos selecionados ficam, em geral, distantes: top 60 passa de 57,75 minutos brutos para 55,98 minutos únicos em `VALIDATION`.
- O híbrido não alcançou 90% de recall direto em `VALIDATION`; por isso não substitui o modo de alta cobertura.
- A auditoria dos três gols que estavam no pool, mas fora do top 60, identificou duas supressões pela diversidade temporal de 45 s e um caso abaixo do top 60 após diversidade. Não foi criada regra específica para esses casos.
- Não retomar detector, tracker, OCR, VLM, novo classificador de vídeo ou novos redutores nesta fase. A próxima etapa é integração reprodutível dos modos congelados em `analyze_match()` e, somente depois, a avaliação final única em `TEST`.

## Evidências

- `GOAL-EVENT-HYBRID-SAFETY-REDUCER-TRAIN-V1/`
- `GOAL-EVENT-WIDE-POOL-MERGE-AUDIT-VALIDATION-V1/`
- `GOAL-EVENT-HYBRID-SAFETY-REDUCER-VALIDATION-CONFIRMATION-V1/`
- `GOAL-EVENT-WIDE-POOL-REDUCTION-VALIDATION-V1/`
