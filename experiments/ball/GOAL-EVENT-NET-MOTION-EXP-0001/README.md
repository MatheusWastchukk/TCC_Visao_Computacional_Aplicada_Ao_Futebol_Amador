# GOAL-EVENT-NET-MOTION-EXP-0001

## Hipótese

Movimento local na região da goleira, após compensação simples da câmera,
poderia complementar o ranking de candidatos próximos ao gol.

## Protocolo

- Escopo exclusivo em `TRAIN`.
- Foram medidos 5.372 grupos de 22 partidas já mineradas, sem reexecutar os
  detectores de bola e estrutura.
- Para cada grupo, o frame representativo foi comparado ao frame 0,2 s antes.
- O sinal mede diferença estabilizada dentro da caixa da goleira, no anel
  externo e a razão entre ambos. Não afirma que a rede se moveu por gol.
- Comparação LOMO pareada com o mesmo Random Forest do baseline e os mesmos
  limites de revisão.

## Resultado LOMO em TRAIN

| K | Baseline | Com movimento de rede | Diferença |
|---:|---:|---:|---:|
| 4 | 14/130 | 15/130 | +1 |
| 8 | 24/130 | 22/130 | -2 |
| 12 | 31/130 | 33/130 | +2 |
| 20 | 48/130 | 46/130 | -2 |
| 30 | 58/130 | 62/130 | +4 |
| 60 | 85/130 | 86/130 | +1 |

## Decisão

**NO-GO.** Os ganhos não são consistentes, a melhora de Top-60 é marginal e a
extração completa levou aproximadamente 25 minutos apenas para o conjunto de
desenvolvimento. O sinal não será integrado ao V4 nem confirmado em
`VALIDATION`; `TEST` permanece fechado.
