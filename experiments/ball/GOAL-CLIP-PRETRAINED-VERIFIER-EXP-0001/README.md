# GOAL-CLIP-PRETRAINED-VERIFIER-EXP-0001

Baseline de verificador visual de clipes com uma ResNet18 pré-treinada no ImageNet e congelada.

## Protocolo

- Escopo exclusivo em `TRAIN`; `VALIDATION` e `TEST` não foram acessados.
- Dataset V2 auditado: 304 clipes de 30 s, 152 `GOAL` e 152 `NO_GOAL`, distribuídos em 25 partidas.
- Oito frames uniformes por clipe; embeddings ResNet18 agregados por média, desvio-padrão, diferença início-fim e variação entre frames (2.048 atributos).
- Classificadores leves avaliados em leave-one-match-out. Não houve ajuste fino da ResNet nem ajuste usando a partida avaliada.

## Resultado global

| Modelo | AP | AUC ROC | Precisão em 0,5 | Recall em 0,5 |
| --- | ---: | ---: | ---: | ---: |
| Regressão logística | 66,21% | 64,61% | 61,49% | 59,87% |
| Floresta aleatória | 66,68% | 64,94% | 62,99% | 52,63% |

O resultado supera o descritor visual artesanal (melhor AP 15,35%; AUC 51,78%) e indica sinal visual transferível entre partidas. A floresta obteve a maior AP; a regressão logística reteve mais recall no limiar 0,5.

## Estratificação de negativos

Para a floresta aleatória, a AUC foi 64,05% contra os 74 negativos difíceis revisados e 65,78% contra os 78 negativos temporais seguros. Essa consistência reduz, mas não elimina, risco de dependência da fonte do negativo.

## Limites e próxima decisão

Este é um experimento de desenvolvimento, não uma avaliação final independente: 19 dos 78 negativos temporais seguros tiveram auditoria humana amostral.

O limiar selecionado em TRAIN para recall mínimo de 80% foi 0,45027083. Ele reteve 84,87% de recall, mas com 68,42% de taxa de falsos positivos e 55,36% de precisão. No limite mais conservador de até 20% de falsos positivos, o melhor recall foi 42,11% (limiar 0,51912500). Portanto, o verificador pré-treinado tem sinal visual útil, mas **não é aceitável como filtro binário operacional nesta versão**.

Decisão: não abrir `VALIDATION` nem usar este limiar no pipeline. A arquitetura congelada pode orientar uma futura hipótese de refinamento temporal, desde que seja definida antes de novo treinamento.

## Artefatos

- `summary_v1.json`: métricas agregadas.
- `predictions_lomo_v1.csv`: previsões fora da partida de treino.
- `threshold_selection_v1/`: varredura e decisão de limiar em TRAIN.
- `clip_embeddings_v1.csv`: embeddings gerados localmente; ignorado pelo Git por ser artefato reproduzível de 7 MB.
