# Como preencher as anotacoes da goleira

## Preparacao

Abra o PowerShell na raiz do projeto e execute:

```powershell
.\.venv\Scripts\python.exe experiments/datasets/annotate_goal_points.py `
  --input experiments/datasets/goal_geometry_annotations_v1.csv `
  --output experiments/datasets/goal_geometry_annotations_v1.csv
```

O programa abrira uma janela para cada um dos 35 frames que possuem
`main_goal` ou `secondary_goal`.

## Ordem dos cliques

Clique nesta ordem:

1. `LT`: topo da trave esquerda, onde ela encontra o travessao;
2. `RT`: topo da trave direita, onde ela encontra o travessao;
3. `LB`: base da trave esquerda;
4. `RB`: base da trave direita.

O programa converte automaticamente os cliques da previa para a resolucao
original da imagem e grava as coordenadas em pixels.

## Controles

- `S` ou `Enter`: salva os quatro pontos e avanca;
- `R`: limpa os pontos do frame atual;
- `N`: pula o frame sem concluir a anotacao;
- `Q` ou `Esc`: salva o progresso e encerra.

## Como interpretar as coordenadas

As coordenadas usam esta convencao:

```text
x = distancia em pixels a partir da borda esquerda
y = distancia em pixels a partir do topo
```

Em uma imagem `1920x1080`, o ponto `(720, 300)` esta 720 pixels a direita e
300 pixels abaixo do canto superior esquerdo.

Se uma parte estiver ocluida, marque a melhor estimativa do endpoint visivel e
registre a situacao em `notes`. Nao invente uma goleira em frames que nao
possuem estrutura; esses frames nao estao neste arquivo de anotacao.

Ao terminar, confirme que `annotation_status` esta como `completed` para os
frames revisados. O arquivo preenchido sera usado para calcular proporcoes,
distancias normalizadas e os scores da V8.

## Anotar os 15 frames externos da V8.5

Para preencher os pontos dos frames externos, use o template:

```text
experiments/datasets/v85_external_goal_annotations.csv
```

Execute:

```powershell
.\.venv\Scripts\python.exe -m experiments.datasets.annotate_goal_points `
  --input experiments/datasets/v85_external_goal_annotations.csv `
  --output experiments/datasets/v85_external_goal_annotations.csv `
  --frames-dir experiments/goal_detection_pilot/data/frames_v85
```

O programa abrira cada JPG em uma janela. Clique nos quatro pontos na ordem
LT, RT, LB e RB. As coordenadas serao calculadas automaticamente; voce nao
precisa medir pixels nem digitar coordenadas.

Para o frame `match_02_004804.833`, use o arquivo
`v85_external_goal_annotations_multitarget.csv`. Ele possui tres linhas para o
mesmo frame, uma para cada estrutura de gol visivel. Para abrir somente essas
linhas pendentes:

```powershell
.\.venv\Scripts\python.exe -m experiments.datasets.annotate_goal_points `
  --input experiments/datasets/v85_external_goal_annotations_multitarget.csv `
  --output experiments/datasets/v85_external_goal_annotations_multitarget.csv `
  --frames-dir experiments/goal_detection_pilot/data/frames_v85 `
  --pending-only
```
