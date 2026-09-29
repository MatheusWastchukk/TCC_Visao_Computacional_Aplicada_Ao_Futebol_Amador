# GOAL-EVENT-SHOT-MOTION-EXP-0001

## Hipótese

Movimento da bola compatível com aproximação rápida e dirigida ao gol pode
ajudar a priorizar candidatos ofensivos. Os atributos não detectam nem
confirmam um chute.

## Protocolo congelado para a hipótese

- Escopo de desenvolvimento: somente `TRAIN`.
- Mesma amostra pareada do baseline: 5.372 grupos de 22 partidas e 130
  timestamps catalogados avaliáveis.
- Modelo: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
  class_weight=balanced, random_state=42)` em leave-one-match-out.
- Evidências acrescentadas: número de observações da trajetória escolhida,
  velocidade média e máxima normalizada, cosseno médio para o centro da
  goleira, fração de aproximação e variação de distância normalizada.
- Não houve ajuste de limiar, janela, detector ou seleção com `VALIDATION` ou
  `TEST`.

## Resultado LOMO em TRAIN

| K | Baseline | Com movimento ofensivo | Diferença |
|---:|---:|---:|---:|
| 4 | 14/130 (10,77%) | 11/130 (8,46%) | -3 |
| 8 | 24/130 (18,46%) | 27/130 (20,77%) | +3 |
| 12 | 31/130 (23,85%) | 31/130 (23,85%) | 0 |
| 20 | 48/130 (36,92%) | 43/130 (33,08%) | -5 |
| 30 | 58/130 (44,62%) | 61/130 (46,92%) | +3 |
| 60 | 85/130 (65,38%) | 90/130 (69,23%) | +5 |

## Decisão de desenvolvimento

O resultado é **promissor somente para o objetivo de alta cobertura** (`K=60`),
que é o modo principal do projeto. Como há perdas em limites menores, o sinal
não substitui os demais modos e não será integrado à aplicação neste momento.

A configuração acima está congelada. A próxima ação permitida é uma única
confirmação em `VALIDATION`, comparando apenas o Top-60 contra o baseline sob
as mesmas janelas. `TEST` permanece fechado.

## Confirmação única em VALIDATION

| Método | Top-60 em VALIDATION |
|---|---:|
| Baseline pareado treinado nas 22 partidas disponíveis | 34/51 (66,67%) |
| Com movimento ofensivo | 35/51 (68,63%) |
| Modo operacional V4 de alta cobertura já congelado | 47/51 (92,16%) |

## Decisão final

**NO-GO.** O atributo acrescentou somente um gol ao baseline experimental
pareado e permaneceu muito abaixo da cobertura do V4 já congelado. Ele não
será integrado ao ranking, à aplicação ou a qualquer modo operacional. Não
serão feitos novos ajustes nessa hipótese a partir de `VALIDATION` e `TEST`
permanece fechado.
