# V10 — Evolução do detector visual de estruturas de gol

## Objetivo

A V10 implementa um detector visual supervisionado para localizar qualquer estrutura de gol visível no frame. A classe é única:

```text
goal_structure
```

`main_goal` e `secondary_goal` não são separados nesta fase. Todas as estruturas reais visíveis são consideradas válidas.

## Dados e anotações

Os vídeos foram processados na resolução original de `1920x1080`. As anotações manuais com `LT`, `RT`, `LB` e `RB` foram convertidas para caixas YOLO.

Para goleiras parcialmente visíveis, foi utilizada somente a extensão visível anotada, sem inventar pontos ocultos.

Dataset ampliado usado no treinamento (`EXP-0067`):

- 256 frames de desenvolvimento;
- 206 estruturas positivas;
- 50 frames negativos de desenvolvimento;
- 3 partidas: `match_01`, `match_02` e `match_03`.

Os 22 novos `not_goal` do primeiro lote não foram usados no treinamento. Eles foram preservados para avaliação externa.

## Estratégia de validação

Foi utilizado leave-one-match-out:

| Fold | Treinamento | Validação interna |
|---|---|---|
| Fold 1 | `match_02` + `match_03` | `match_01` |
| Fold 2 | `match_01` + `match_03` | `match_02` |
| Fold 3 | `match_01` + `match_02` | `match_03` |

Essa separação evita misturar frames da mesma partida entre treinamento e validação.

## Pipeline implementado

1. Preparação das anotações e conversão para YOLO.
2. Materialização dos conjuntos `train` e `val` por partida.
3. Treinamento do YOLO11n com pesos pré-treinados.
4. Avaliação externa em frames positivos e `not_goal`.
5. Cálculo de recall por IoU, IoU médio e falsos positivos.
6. Geração de overlays com previsão e anotação real.
7. Calibração do limiar usando somente a validação interna.

O avaliador considera uma previsão correta quando a melhor caixa possui IoU maior ou igual a `0,30` ou `0,50`, conforme a métrica analisada.

## Resultados antes da ampliação

Na primeira rodada V10, com 320 px e dataset menor, os folds apresentaram:

```text
33,3% / 93,3% / 26,7%
```

Essa grande variação mostrou que o detector aprendia visualmente, mas não generalizava de forma estável entre partidas.

## Ampliação do dataset

Foram preparados e anotados dois lotes adicionais:

- primeiro lote: 75 frames, com 53 positivos e 22 `not_goal`;
- segundo lote: 180 frames, com 123 positivos e 57 `not_goal`.

Os novos positivos foram incorporados ao dataset de desenvolvimento. Os negativos foram mantidos para testar falsos positivos.

Ao todo, a avaliação externa passou a conter 94 frames `not_goal`:

- 38 em `match_01`;
- 27 em `match_02`;
- 29 em `match_03`.

A meta formal de 30 negativos por partida ainda não foi atingida em `match_02` e `match_03`, mas o conjunto já é significativamente maior que o inicial.

## Calibração do limiar

Foram testados os limiares `0,01`, `0,05`, `0,10` e `0,25` nas validações internas.

O valor provisoriamente escolhido foi:

```text
confidence = 0,05
```

O limiar `0,01` aumentou o recall, mas também produziu muitas previsões indevidas. O limiar `0,25` foi excessivamente restritivo para os modelos ampliados.

## Resultado com 320 px

Com YOLO11n, 10 épocas, `imgsz=320` e `confidence=0,05`:

| Experimento | Fold | Recall IoU ≥ 0,50 | Falsos positivos |
|---|---|---:|---:|
| EXP-0078 | sem `match_03` | 80,0% | 0/94 |
| EXP-0079 | sem `match_01` | 73,3% | 0/94 |
| EXP-0080 | sem `match_02` | 66,7% | 0/94 |

Resumo:

```text
media: 73,3%
minimo: 66,7%
maximo: 80,0%
amplitude: 13,3 pontos percentuais
```

## Experimento com 640 px

Foi repetido o mesmo treinamento, alterando somente a resolução de entrada para `640x640`:

- YOLO11n;
- dataset EXP-0067;
- 10 épocas;
- CPU;
- mesmo esquema de folds;
- mesmo limiar externo `0,05`.

Resultados externos:

| Experimento | Fold | Recall IoU ≥ 0,50 | Falsos positivos |
|---|---|---:|---:|
| EXP-0085 | sem `match_03` | 86,7% | 0/94 |
| EXP-0086 | sem `match_01` | 100,0% | 0/94 |
| EXP-0087 | sem `match_02` | 93,3% | 2/94 |

Resumo:

```text
media: 93,3%
minimo: 86,7%
maximo: 100,0%
amplitude: 13,3 pontos percentuais
falsos positivos: 2/94
```

### Comparação entre resoluções

| Resolução | Recall médio | Pior fold | Melhor fold | Falsos positivos |
|---:|---:|---:|---:|---:|
| 320 px | 73,3% | 66,7% | 80,0% | 0/94 |
| 640 px | 93,3% | 86,7% | 100,0% | 2/94 |

Com o mesmo dataset e número de épocas, a resolução de 640 px aumentou substancialmente o recall. A hipótese é que estruturas pequenas, distantes ou formadas por barras finas preservam mais informação em 640 px.

## Análise dos falsos positivos em 640 px

Os dois falsos positivos ocorreram no fold `EXP-0087`, treinado sem `match_02` e validado em `match_03`:

- [match_03_004538.066.jpg](EXP-0087/overlays/match_03/match_03_004538.066.jpg)
- [match_03_003816.489.jpg](EXP-0087/overlays/match_03/match_03_003816.489.jpg)

Em ambos os casos, o modelo marcou uma estrutura retangular formada por casas/vegetação ao fundo, sem uma goleira real. As previsões tinham confiança próxima do limiar e IoU igual a zero.

Esses overlays são importantes porque mostram que o erro residual não é mais predominantemente ausência de detecção, mas confusão visual localizada em elementos retangulares do cenário.

## Experimentos principais

| Experimento | Finalidade |
|---|---|
| EXP-0045 a EXP-0049 | Preparação e ampliação dos datasets V10 |
| EXP-0050 | Smoke test do treinamento YOLO |
| EXP-0051 a EXP-0055 | Primeiros treinamentos e avaliações dos folds |
| EXP-0057 a EXP-0059 | Avaliação positiva e negativa inicial |
| EXP-0060 | Resumo da validação positiva/negativa |
| EXP-0061 | Primeiro lote de hard negatives |
| EXP-0062 a EXP-0064 | Avaliação com hard negatives do primeiro lote |
| EXP-0065 | Consolidação da avaliação com 37 negativos |
| EXP-0066/EXP-0067 | Dataset ampliado com novos positivos |
| EXP-0068 a EXP-0070 | Treinamento ampliado em 320 px |
| EXP-0071 a EXP-0073 | Avaliação e varredura de confiança |
| EXP-0074/EXP-0075 | Calibração exploratória e interna do limiar |
| EXP-0076 | Segundo lote de hard negatives |
| EXP-0078 a EXP-0080 | Validação ampliada em 320 px |
| EXP-0082 a EXP-0084 | Treinamento em 640 px |
| EXP-0085 a EXP-0087 | Avaliação externa dos folds 640 px |
| EXP-0088 | Comparação formal entre 320 e 640 px |

## Interpretação atual

A V10 demonstrou que o problema é aprendível visualmente e que a resolução de entrada tem impacto relevante. A configuração 640 px apresentou o melhor resultado até agora, com recall médio de 93,3% e baixa variação entre partidas.

Entretanto, ainda não é correto afirmar que o detector está pronto para vídeos completos, porque:

- o conjunto externo ainda é pequeno em frames positivos;
- `match_02` e `match_03` ainda não possuem exatamente 30 negativos;
- há dois falsos positivos em hard negatives;
- ainda não foi feita avaliação temporal em sequência contínua;
- o desempenho foi medido em três partidas.

## Próximos passos

1. Revisar e categorizar os dois falsos positivos do `EXP-0087`.
2. Completar pelo menos 30 negativos em `match_02` e `match_03`.
3. Reexecutar a avaliação com `imgsz=640` e `confidence=0,05` congelados.
4. Calcular métricas finais por partida: recall, precisão, IoU médio, TN e FP.
5. Avaliar falsos positivos por minuto em uma sequência de vídeo.
6. Gerar overlays de verdadeiros positivos, falsos negativos e falsos positivos.
7. Se a estabilidade se mantiver, iniciar a V9 temporal para suavização e associação das detecções.

A V9 não deve substituir a localização espacial. Ela será usada depois da V10 para persistência, associação e redução de falhas isoladas entre frames.

## Arquivos de referência

- `prepare_v10_dataset.py`: prepara o dataset YOLO.
- `materialize_v10_yolo.py`: cria os folds de treinamento e validação.
- `evaluate_v10_detector.py`: avalia positivos e negativos e gera overlays.
- `calibrate_v10_threshold.py`: calibra o limiar nas validações internas.
- `EXP-0067/summary.json`: resumo do dataset ampliado.
- `EXP-0075/internal_threshold_sweep.json`: calibração interna do limiar.
- `EXP-0088/comparison_320_vs_640.json`: comparação entre resoluções.
