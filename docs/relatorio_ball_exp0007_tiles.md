# Relatório do `BALL-EXP-0007-TILES`

## Execução

- Dataset: tiles 2x2 derivados dos mesmos frames e splits.
- Total: 3.408 tiles, sendo 2.932 para treino e 476 para validação.
- Tiles positivos: 709.
- Modelo: YOLO11n, 10 épocas, `imgsz=960`, `batch=8`.
- Hardware: AMD Radeon RX 9060 XT via ROCm, sem AMP.
- Tempo: aproximadamente 31,2 minutos.

## Métricas internas dos tiles

Precisão: 21,43%; recall: 16,67%; mAP50: 13,13%; mAP50-95: 4,47%.
Essas métricas não são diretamente comparáveis às anteriores, pois foram
calculadas sobre imagens de tiles.

## Avaliação nos frames originais

Cada um dos 119 frames de validação foi dividido em quatro tiles. As caixas
foram remapeadas para as coordenadas do frame completo e combinadas por NMS
global.

| Resultado | Quantidade |
|---|---:|
| TP | 6 |
| FN | 67 |
| TN | 46 |
| FP | 0 |

Nos 73 frames com bola localizável (`VISIBLE`/`PARTIAL`), o recall automático
preliminar foi `6/73 = 8,22%`. Esse resultado ainda não é conclusivo: os
overlays precisam ser revisados visualmente, pois o critério automático pode
classificar como FN uma caixa visualmente aceitável com IoU abaixo de 0,5.

## Artefatos

- Avaliador: `experiments/ball/audit_ball_tiles.py`.
- Pesos: `runs/detect/experiments/ball/BALL-EXP-0007-TILES/weights/best.pt`.
- Auditoria: `experiments/ball/BALL-EXP-0007-TILES/audit_tiles_v1/`.

## Revisão manual concluída

Os 119 overlays foram revisados manualmente:

| Visibilidade | Correto | Falso negativo |
|---|---:|---:|
| `VISIBLE` | 7 | 60 |
| `PARTIAL` | 2 | 4 |
| `NOT_VISIBLE` | 46 | 0 |

Considerando somente os 73 frames localizáveis, foram 9 corretos e 64 falsos
negativos, resultando em recall manual de `9/73 = 12,33%`. Não foram marcados
falsos positivos nem inconclusivos.

Conclusão preliminar: o recorte 2x2 não demonstrou ganho sobre o modelo
principal de 960 px, que havia alcançado 41,10% de recall manual. O piloto de
tiles deve permanecer como comparação negativa, e a próxima etapa deve focar
em melhorar a representação dos objetos pequenos, especialmente blur,
oclusão e bola nas mãos do goleiro, antes de tentar integrar tracking.

## Comparação justa com 30 épocas

O `BALL-EXP-0008-TILES-30` repetiu o treinamento com 30 épocas, mantendo o
dataset e a configuração GPU. A inferência tiled produziu inicialmente 13 TP,
60 FN, 44 TN e 2 FP. Após a revisão manual dos 119 overlays, o resultado foi:

- 73 frames localizáveis: 19 corretos e 54 falsos negativos;
- recall manual localizável: `19/73 = 26,03%`;
- 46 frames `NOT_VISIBLE`: 45 corretos e 1 falso negativo revisado;
- falsos positivos: 0;
- inconclusivos: 0.

O treinamento mais longo melhorou o recall manual de 12,33% (`EXP-0007`) para
26,03%, mas ainda ficou abaixo dos 41,10% do `BALL-EXP-0005` sem tiles. Assim,
os tiles podem recuperar parte das bolas pequenas, porém não devem substituir
o baseline principal neste momento.
