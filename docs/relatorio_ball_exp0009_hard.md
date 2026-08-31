# Relatório `BALL-EXP-0009-HARD-30`

## Configuração

- Modelo: YOLO11n pré-treinado.
- Dataset: `yolo_hard_v1`, com 995 imagens de treino e 119 de validação.
- Reforço: 131 imagens difíceis identificadas pelo `EXP-0005`, duplicadas duas vezes.
- Resolução: 960 px.
- Épocas: 30.
- Hardware: AMD Radeon RX 9060 XT via ROCm, sem AMP.
- Tempo: aproximadamente 31,7 minutos.

## Métricas automáticas de validação

- Precisão: 36,79%.
- Recall: 39,07%.
- mAP50: 33,53%.
- mAP50-95: 9,60%.

## Auditoria manual

Os 119 overlays foram revisados. A distribuição foi:

| Visibilidade | Correto | Falso negativo | Falso positivo |
|---|---:|---:|---:|
| `VISIBLE` | 25 | 42 | 0 |
| `PARTIAL` | 0 | 6 | 0 |
| `NOT_VISIBLE` | 42 | 1 | 3 |

Entre os 73 frames localizáveis (`VISIBLE`/`PARTIAL`), foram 25 corretos e
48 falsos negativos. O recall manual localizável foi `25/73 = 34,25%`.

Quatro frames foram mantidos como falso negativo porque o modelo encontrou uma
bola diferente da bola-alvo: frames 8, 18, 27 e 99 da auditoria. Esses casos
devem ser contabilizados como perda da bola-alvo e também como detecção
incorreta na análise de erros.

## Comparação

O recall manual de 34,25% ficou abaixo do `BALL-EXP-0005`, que alcançou 41,10%
com o dataset original. Portanto, a sobreamostragem dos 131 exemplos difíceis
melhorou a capacidade automática em relação ao baseline, mas não demonstrou
ganho na auditoria manual principal.

## Próximas etapas

1. Não substituir o `EXP-0005` pelo `EXP-0009` como modelo principal.
2. Calibrar o limiar de confiança usando os mesmos overlays, procurando
   recuperar detecções sem aumentar excessivamente as detecções de bolas erradas.
3. Se o limiar não resolver os quatro casos de bola incorreta, selecionar novos
   exemplos de treino especificamente desses contextos, sem usar a validação.
4. Depois avaliar o melhor modelo em trechos contínuos de vídeo antes de iniciar
   tracking ou integração com a detecção de gols.

## Calibração do limiar de confiança

Foram avaliados cinco limiares usando os mesmos 119 frames e o mesmo modelo.
Os números abaixo são automáticos e usam 73 frames localizáveis:

| Confiança | Recall | Precisão | TP | FP |
|---:|---:|---:|---:|---:|
| 0,10 | 32,88% | 58,54% | 24 | 17 |
| 0,15 | 31,51% | 67,65% | 23 | 11 |
| 0,20 | 26,03% | 70,37% | 19 | 8 |
| 0,25 | 23,29% | 77,27% | 17 | 5 |
| 0,30 | 21,92% | 88,89% | 16 | 2 |

O limiar 0,10 recupera mais detecções, mas gera muitos falsos positivos e
ainda não ultrapassa o desempenho manual do `EXP-0005`. O limiar 0,25 foi
mantido como compromisso entre recall e precisão para os próximos testes.
