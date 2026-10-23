# Protocolo pré-registrado da avaliação final TEST V3

**Registrado em:** 19/10/2026, antes da inferência TEST.<br>
**Estado:** protocolo fixo; ainda sem resultados preditivos.<br>
**Freeze operacional:** `experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v3.json`.<br>
**Catálogo humano:** `goal_catalog_timestamp_manifest_frozen_v1.csv`; SHA-256 `593fabd777d0e0cf99f723d7a3853d7c8c5d08e951ce54224300204fd824f1b4`.

## Pergunta e população

Medir, uma única vez, a cobertura dos timestamps humanos de gol pelo vídeo entregue por cada uma das três políticas V3 congeladas. A população é composta pelos nove vídeos do split `test` em `experiments/datasets/dataset_split_v1.csv` e pelos 57 registros humanos `completed`. O operador confirmou que anotou os vídeos originais e o catálogo-base sem consultar saídas da IA.

O resultado mede **cobertura de eventos catalogados**, não precisão automática de gols. O sistema é um minerador/priorizador de trechos para revisão humana.

## Políticas imutáveis

| Política | Pontuação/RRF | Janela de entrega | K | Separação mínima |
|---|---|---:|---:|---:|
| `FAST_V3` | `RRF4_NET` | 30 s (10 s antes, 20 s depois) | 60 | 45 s |
| `BALANCED_V3` | `RRF4_NET` | 40 s (13,333 s antes, 26,667 s depois) | 60 | 45 s |
| `HIGH_RECALL_V3` | `RRF4_NET` | 60 s (20 s antes, 40 s depois) | 55 | 45 s |

Fontes do RRF4_NET, em ordem fixa: `rank_score`, `visual_score_goal`, `fusion_score_goal`, `net_motion`; para cada fonte, ordenar por escore decrescente e `group_uid` crescente em empate; contribuição recíproca `1 / (60 + rank)`. A política percorre essa ordem e rejeita grupos cujo centro esteja a menos de 45 s de um já selecionado. Não há rescue, busca de hiperparâmetros, escolha de vencedor ou modo V4 comparador.

## Geração de candidatos e pontuação

O ranqueamento recebe exclusivamente os vídeos originais, manifesto de split, detectores GOAL/BALL, bundle de quatro modelos treinados em TRAIN e código de extração de atributos. Não lê o catálogo TEST, timestamps humanos, saídas preditivas anteriores, rótulos, rankings, shortlist ou clipes do TEST. O catálogo congelado será aberto somente em uma etapa separada de cálculo de métricas, depois que a lista completa de candidatos, escores, seleções e versões da inferência estiver gravada.

Os candidatos são todos os grupos temporais elegíveis produzidos pelo minerador congelado: amostragem de 5 FPS, confiança GOAL 0,25, confiança BALL 0,05, resolução GOAL 640 e BALL 960, sem crop ROI, distância normalizada máxima 2, margem da estrutura 0,35, gap de grupo 3 s, mínimo de 2 hits e separação de mineração 45 s. O limite de 60 grupos controla o artefato de candidatos selecionados pelo minerador; a tabela de frames mantém os hits usados para reconstruir o pool completo. Confirmar hashes dos detectores e registrar todas as opções efetivas antes do processamento.

Os quatro modelos V3 devem corresponder exatamente aos hashes de `model_manifest_v3.json`. Os atributos temporais, net-motion e embeddings visuais devem seguir os contratos usados pelo pacote REPORT94; registrar hashes do código executado, dependências relevantes e pesos do encoder visual. O bundle de modelos é somente inferido; nenhum ajuste ou fit com dados TEST é permitido.

### Integridade de entrada conferida antes da execução

| Entrada | SHA-256 |
|---|---|
| GOAL EXP-0092 `best.pt` | `76ebf39e7b6d76ff69b5f644fbb7d1d2c742c79bd5154af245cb6bd417bba5d1` |
| BALL EXP-0005 `best.pt` | `c7368e56ce8564a31d484312102b8aa056189307168a50a276757861a60a2615` |
| ResNet18 ImageNet `resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Temporal ranker | `08c1aeeae283284d1128b59deb94a25738312e23fc8f08518591f7a71fcdfbac` |
| Net-motion ranker | `6b59901e214d8854beee20c80018db77fe272f15b3895f295c4fc2040d01e87c` |
| Visual verifier | `c4b31c8cab60d0af0bbd28f7284d7bb7958dbdecd3c93d5d4f84998949d87a1d` |
| Fusion ranker | `cd1228d242a3a1a53e39f1866d630e6640d477c3824286010f008fb9ba839b11` |

O commit-base do checkout é `92a080dab4fc136edc6430f5e15366d1b0f4d1b7`, igual ao `candidate_head_sha` histórico do freeze V3. Há alterações locais no checkout; o hash do script e dos auxiliares realmente usados será gravado no manifesto de execução.

## Métricas pré-definidas

- **Primária:** `goal_covered` se o timestamp humano pertencer a pelo menos um intervalo selecionado; usar inclusão fechada (`início <= timestamp <= fim`) e tolerância adicional igual a zero.
- **Cobertura geral:** gols cobertos / 57; além do agregado, apresentar gols avaliáveis, cobertos e perdidos por partida e modo.
- **Carga de vídeo:** união dos intervalos selecionados, após limitar cada intervalo aos limites reais do arquivo `[0, duração]`; relatar minutos únicos e `100 × minutos únicos / duração original` por partida e agregado. O corte no limite do arquivo só remove tempo inexistente e não afeta ranking ou seleção.
- **Volume:** número de candidatos elegíveis, candidatos selecionados e segmentos após unir intervalos sobrepostos ou encostados, por partida e modo.
- **Tempo computacional:** duração de mineração, extração de atributos/embeddings, pontuação e seleção por partida e total; registrar hardware e dispositivo efetivo.
- **Comparação pareada:** ganhos/perdas de gols cobertos entre cada par de modos, sem declarar vencedor nem alterar políticas.
- **Sensibilidade:** exibir resultados por partida; qualquer análise de partidas com exposição histórica será descritiva e não poderá alterar o protocolo ou as políticas.

## Falhas e preservação

Preservar pool completo, escores, seleções, intervalos, logs, tempos, hashes e erros. Falha de infraestrutura pode ser repetida somente com os mesmos arquivos, código, parâmetros e dispositivo, registrando tentativa inicial e repetição. Não retreinar, não retunar e não reiniciar a avaliação para tentar outra configuração depois de ver qualquer resultado TEST.
