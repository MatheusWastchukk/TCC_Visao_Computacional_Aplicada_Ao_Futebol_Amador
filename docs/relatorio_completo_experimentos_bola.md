# Relatório completo dos experimentos de detecção da bola

**Projeto:** detecção e análise de vídeos de futebol  
**Escopo:** detector frame a frame da bola  
**Data de consolidação:** 27/08/2026

## 1. Objetivo

Esta etapa teve como objetivo construir e avaliar um detector de bola para
vídeos de futebol. O detector foi baseado no YOLO11n pré-treinado e treinado
com imagens extraídas dos vídeos do projeto.

O foco inicial foi medir a viabilidade da detecção frame a frame antes de
integrar o modelo a rastreamento temporal ou à lógica de detecção de gols.

## 2. Protocolo de anotação

Foram definidos quatro estados para a visibilidade da bola:

- `VISIBLE`: a bola está visível e sua posição pode ser marcada com uma caixa;
- `PARTIAL`: a bola está parcialmente ocultada, borrada ou cortada, mas ainda
  é possível indicar sua região;
- `NOT_VISIBLE`: a bola está fora do frame ou não há evidência visual
  suficiente;
- `UNCERTAIN`: existe uma região suspeita, mas não é possível confirmar a
  presença da bola com segurança.

Imagens `NOT_VISIBLE` foram mantidas como imagens negativas, sem caixa. As
imagens `UNCERTAIN` foram excluídas da avaliação principal para evitar que
casos sem confirmação visual fossem tratados como erro do detector.

As anotações foram realizadas com o aplicativo de marcação de bounding boxes,
com salvamento incremental e retomada por `--pending-only`.

## 3. Construção dos datasets

### 3.1 BALL-DATASET-V1

O primeiro lote foi composto por 620 frames anotados:

| Visibilidade | Quantidade |
|---|---:|
| `VISIBLE` | 395 |
| `PARTIAL` | 25 |
| `NOT_VISIBLE` | 184 |
| `UNCERTAIN` | 16 |
| **Total** | **620** |

A seleção original foi feita com seed `20260826`, mantendo partidas separadas
entre treino e validação. O conjunto `test` permaneceu fechado.

Após a conversão para o formato YOLO, foram utilizados:

- treino: 485 imagens, incluindo 347 imagens positivas/localizáveis e 138
  imagens negativas;
- validação: 119 imagens, incluindo 73 imagens positivas/localizáveis e 46
  imagens negativas;
- 16 imagens `UNCERTAIN` excluídas.

### 3.2 Ampliação para BALL-DATASET-V2

Foi anotado um lote adicional de 250 frames:

| Visibilidade | Quantidade |
|---|---:|
| `VISIBLE` | 177 |
| `PARTIAL` | 8 |
| `NOT_VISIBLE` | 63 |
| `UNCERTAIN` | 2 |
| **Total** | **250** |

O dataset combinado passou a ter 870 anotações:

| Visibilidade | Quantidade |
|---|---:|
| `VISIBLE` | 572 |
| `PARTIAL` | 33 |
| `NOT_VISIBLE` | 247 |
| `UNCERTAIN` | 18 |
| **Total** | **870** |

Na preparação YOLO v2 foram incluídas 852 imagens e excluídas as 18 imagens
`UNCERTAIN`:

- treino: 733 imagens, sendo 532 positivas/localizáveis e 201 negativas;
- validação: 119 imagens, sendo 73 positivas/localizáveis e 46 negativas.

A validação foi mantida igual entre os experimentos para permitir comparação
direta. Os principais arquivos são:

- `experiments/ball/ball_dataset_annotations_v1.csv`;
- `experiments/ball/ball_dataset_annotations_v2.csv`;
- `experiments/ball/ball_dataset_v1.yaml`;
- `experiments/ball/ball_dataset_v2.yaml`.

## 4. Experimentos realizados

Todos os experimentos utilizaram YOLO11n pré-treinado para uma única classe,
`ball`, com resolução principal de 640 px e lote 8, salvo indicação contrária.

### 4.1 BALL-EXP-0001 — baseline inicial

- dataset: `BALL-DATASET-V1`;
- épocas: 10;
- dispositivo: CPU;
- duração: 1116 s, aproximadamente 18,6 minutos.

| Métrica | Resultado |
|---|---:|
| Precisão | 24,60% |
| Recall | 15,10% |
| mAP50 | 11,60% |
| mAP50-95 | 4,42% |

Auditoria manual dos 119 frames:

- correto: 54;
- falso negativo: 64;
- falso positivo: 0;
- inconclusivo: 1;
- recall em frames localizáveis: 10/73 = 13,70%.

O resultado confirmou que o modelo inicial era insuficiente para iniciar o
rastreamento.

### 4.2 BALL-EXP-0002 — treinamento prolongado

- dataset: `BALL-DATASET-V1`;
- épocas: 30;
- dispositivo: CPU;
- duração: 3793 s, aproximadamente 63,2 minutos.

| Métrica | EXP-0001 | EXP-0002 |
|---|---:|---:|
| Precisão | 24,60% | 29,26% |
| Recall | 15,10% | 28,77% |
| mAP50 | 11,60% | 21,19% |
| mAP50-95 | 4,42% | 7,08% |

Auditoria manual:

- correto: 61;
- falso negativo: 57;
- falso positivo: 0;
- inconclusivo: 1;
- recall em frames localizáveis: 16/73 = 21,92%.

O aumento para 30 épocas melhorou todas as métricas principais, mas o recall
continuou baixo para uso seguro no rastreamento.

### 4.3 BALL-EXP-0003 — dataset ampliado

- dataset: `BALL-DATASET-V2`;
- épocas: 30;
- dispositivo: CPU;
- duração: 5305 s, aproximadamente 88,4 minutos.

| Métrica | Resultado |
|---|---:|
| Precisão | 55,96% |
| Recall | 26,03% |
| mAP50 | 27,25% |
| mAP50-95 | 7,85% |

Auditoria manual:

- correto: 66;
- falso negativo: 53;
- falso positivo: 0;
- inconclusivo: 0;
- recall em frames localizáveis: 21/73 = 28,77%.

Este foi o melhor resultado de qualidade até o momento. O aumento do dataset
melhorou especialmente a precisão, o mAP50 e o recall manual, mas ainda não
resolveu a dificuldade de detectar bolas pequenas, distantes ou parcialmente
ocultadas.

### 4.4 BALL-EXP-0004 — primeiro treinamento ROCm/GPU

- dataset: `BALL-DATASET-V2`;
- épocas: 30;
- resolução: 640 px;
- lote: 8;
- dispositivo: AMD Radeon RX 9060 XT via ROCm 7.2.1;
- AMP: desativado devido a instabilidade numérica no smoke test;
- duração: 1355 s, aproximadamente 22,6 minutos.

| Métrica | Resultado |
|---|---:|
| Precisão | 45,91% |
| Recall | 26,75% |
| mAP50 | 25,02% |
| mAP50-95 | 7,12% |

Auditoria manual:

- correto: 65;
- falso negativo: 54;
- falso positivo: 0;
- inconclusivo: 0;
- recall em frames localizáveis: 19/73 = 26,03%.

O recall manual ficou próximo do recall automático. O experimento não superou
o `BALL-EXP-0003` em qualidade, mas demonstrou que o treinamento completo via
GPU AMD é funcional e muito mais rápido.

## 5. Comparação consolidada

| Experimento | Ambiente | Épocas | Tempo | Precisão | Recall | mAP50 | mAP50-95 | Recall manual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EXP-0001 | CPU | 10 | 18,6 min | 24,60% | 15,10% | 11,60% | 4,42% | 13,70% |
| EXP-0002 | CPU | 30 | 63,2 min | 29,26% | 28,77% | 21,19% | 7,08% | 21,92% |
| EXP-0003 | CPU | 30 | 88,4 min | **55,96%** | 26,03% | **27,25%** | **7,85%** | **28,77%** |
| EXP-0004 | ROCm/GPU | 30 | 22,6 min | 45,91% | **26,75%** | 25,02% | 7,12% | 26,03% |

### Interpretação

1. O treinamento prolongado foi melhor que o baseline de 10 épocas.
2. A ampliação do dataset produziu o melhor modelo geral (`EXP-0003`).
3. O treinamento ROCm/GPU não aumentou a qualidade em relação ao melhor
   modelo CPU, pois o ambiente de hardware não muda, por si só, os dados ou a
   estratégia de treinamento.
4. O `EXP-0004` foi aproximadamente 3,9 vezes mais rápido que o `EXP-0003`:
   1355 s contra 5305 s.
5. O recall manual permaneceu abaixo do necessário para considerar o detector
   pronto para rastreamento sem uma etapa adicional de tratamento temporal.

## 6. Configuração da GPU AMD

O ambiente CPU original `.venv` foi preservado para reprodutibilidade. Também
foram testadas duas rotas para a GPU AMD:

### DirectML

O ambiente `.venv-dml` validou operações simples do PyTorch DirectML e
inferência ONNX com `DmlExecutionProvider`. Entretanto, o treinamento
Ultralytics via PyTorch-DirectML falhou por operadores ausentes ou
incompatíveis no backend, incluindo `unique(..., return_counts=True)` e
operações de `scatter` usadas pelo loss.

Essa rota permanece adequada para inferência ONNX experimental, mas não foi
usada para treinamento.

### ROCm

Foi instalado o ambiente `.venv-rocm` com Python 3.12, ROCm 7.2.1, PyTorch
`2.9.1+rocm7.2.1`, torchvision e Ultralytics. O Visual Studio Build Tools 2022
com workload C++ foi instalado para permitir a compilação dos kernels MIOpen.

O PyTorch reconheceu a GPU como `cuda:0`. Um smoke test de tensor e uma
convolução simples foram executados com sucesso. O primeiro smoke test de
treinamento apresentou `NaN` com AMP ativado. Com AMP desativado e o cache do
MIOpen redirecionado para `.rocm_cache`, o smoke test completou:

- 1 época;
- 320 px;
- batch 2;
- aproximadamente 145 s;
- sem `NaN`.

Depois disso, o `BALL-EXP-0004` completou o treinamento real via ROCm.

## 7. Limitações observadas

- A bola ocupa poucos pixels em muitos frames de 640 px.
- Há oclusões por jogadores e goleiros.
- A bola pode estar borrada, nas mãos do goleiro ou fora da região visível.
- Parte dos frames anotados como `NOT_VISIBLE` não permite avaliar a capacidade
  de localização, apenas a ausência de uma detecção indevida.
- O conjunto de validação tem 73 frames localizáveis, portanto o recall manual
  deve ser interpretado como evidência preliminar.
- As auditorias foram realizadas em frames amostrados, não em todos os frames
  contínuos das partidas.
- Os resultados não devem ser usados para afirmar desempenho temporal ou
  detecção de gols.

## 8. Decisão metodológica atual

### Escolha da resolução principal

Após as auditorias manuais, a comparação entre as resoluções ficou:

| Resolução | Experimento | Precisão | Recall automático | mAP50 | mAP50-95 | Recall manual | Tempo |
|---:|---|---:|---:|---:|---:|---:|---:|
| 640 px | BALL-EXP-0004 | 45,91% | 26,75% | 25,02% | 7,12% | 26,03% | 22,6 min |
| 960 px | BALL-EXP-0005 | **65,15%** | 34,25% | 33,84% | 9,62% | **41,10%** | 35,4 min |
| 1280 px | BALL-EXP-0006 | 38,53% | **38,36%** | **36,61%** | **10,62%** | 39,73% | 43,6 min |

Para a etapa seguinte, 960 px foi escolhida como configuração principal. Ela
apresentou o melhor recall manual e a maior precisão, além de custo menor que
1280 px. O modelo de 1280 px será mantido como resultado comparativo, pois
obteve as melhores métricas automáticas, mas não será a configuração padrão
sem uma nova validação.

### Contexto dos falsos negativos no EXP-0005

Foi realizada uma classificação manual da causa principal dos 51 falsos
negativos. Considerando apenas os 43 falsos negativos em frames localizáveis
(`VISIBLE`/`PARTIAL`), a distribuição foi:

| Contexto principal | Quantidade |
|---|---:|
| `BORRADA_BAIXA_QUALIDADE` | 15 |
| `MUITO_PEQUENA` | 10 |
| `OCULTA_JOGADOR` | 9 |
| `COBRANCA_LATERAL` | 6 |
| `MAOS_GOLEIRO` | 2 |
| `OUTRO` | 1 |
| **Total localizável** | **43** |

Também houve 8 falsos negativos em frames originalmente classificados como
`NOT_VISIBLE`, nos quais foi desenhada uma caixa manual durante a revisão.
Esses casos devem ser tratados como revisão da qualidade da anotação e não
misturados ao recall principal dos 73 frames localizáveis.

O resultado indica que o principal gargalo não é exclusivamente a resolução:
qualidade/blur, escala reduzida e oclusão somam a maior parte dos erros. Os
casos de cobrança lateral e bola nas mãos confirmam que o contexto visual
também altera a aparência da bola, embora sejam menos frequentes.

O `BALL-EXP-0003` deve permanecer como baseline de qualidade. O ambiente
ROCm/GPU deve ser usado nos próximos experimentos para reduzir o tempo de
execução, sempre mantendo os mesmos splits e registrando as alterações de
configuração.

O detector ainda não deve ser integrado diretamente ao rastreamento como se
estivesse validado. Antes disso, é necessário testar estratégias voltadas a
objetos pequenos e avaliar o comportamento em vídeo contínuo.

## 9. Próximas etapas

O experimento `BALL-EXP-0005` foi executado posteriormente a este relatório,
com 960 px via ROCm, e obteve precisão de 65,15%, recall de 34,25%, mAP50 de
33,84% e mAP50-95 de 9,62%. A auditoria manual foi concluída com 68 corretos,
51 falsos negativos, 0 falsos positivos e 0 inconclusivos. O recall manual nos
73 frames localizáveis foi 30/73 = 41,10%.

1. Usar `BALL-EXP-0003` como referência e o ambiente ROCm para treinamento.
2. Testar maior resolução ou estratégia de recorte/amostragem para a bola.
3. Comparar as alterações mantendo a validação congelada.
4. Auditar visualmente cada novo modelo, separando `VISIBLE`, `PARTIAL` e
   `NOT_VISIBLE`.
5. Avaliar falsos positivos por vídeo e por hora de vídeo.
6. Somente depois testar rastreamento temporal e integração com a detecção de
   gols.

Foi executado posteriormente o `BALL-EXP-0006` em 1280 px via ROCm. O modelo
obteve precisão de 38,53%, recall de 38,36%, mAP50 de 36,61% e mAP50-95 de
10,62%. A auditoria visual foi gerada para 119 frames, com triagem automática
de 22 TP, 51 FN, 3 FP e 43 TN. A revisão manual foi concluída com 72 corretos,
47 falsos negativos, 0 falsos positivos e 0 inconclusivos; o recall manual foi
29/73 = 39,73% nos frames localizáveis.

## 10. Artefatos principais

- Pesos do melhor modelo atual: `runs/detect/experiments/ball/BALL-EXP-0003/weights/best.pt`;
- pesos do treinamento GPU: `runs/detect/experiments/ball/BALL-EXP-0004/weights/best.pt`;
- métricas: `experiments/ball/BALL-EXP-0001/metrics.json` até
  `experiments/ball/BALL-EXP-0004/metrics.json`;
- auditorias: `experiments/ball/BALL-EXP-0001/audit_v1/` até
  `experiments/ball/BALL-EXP-0004/audit_v1/`;
- runner: `experiments/ball/run_ball_exp0001.py`;
- configuração da GPU: `docs/configuracao-gpu-amd-directml.md`.
