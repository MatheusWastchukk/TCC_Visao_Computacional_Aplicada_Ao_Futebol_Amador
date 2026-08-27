# BALL-DATASET-V1 — Guia de anotação

## Lote preparado

- `500` frames de partidas `train` — 20 por partida;
- `120` frames de partidas `validation` — 15 por partida;
- partidas `test` não foram incluídas;
- seleção aleatória reprodutível com seed `20260826`.

## Aplicativo

Execute a partir da raiz:

```powershell
.\.venv\Scripts\python.exe experiments\ball\annotate_ball_bbox.py `
  --input experiments\ball\ball_dataset_candidates_v1.csv `
  --output experiments\ball\ball_dataset_annotations_v1.csv `
  --pending-only
```

## Controles

- arrastar com o botão esquerdo: desenhar a caixa da bola;
- `V`: bola visível, com caixa obrigatória;
- `P`: bola parcialmente visível, com caixa obrigatória;
- `N`: bola não visível ou fora do frame, sem caixa;
- `U`: situação incerta, sem caixa obrigatória;
- `S` ou `Enter`: salvar e avançar;
- `R`: limpar a caixa;
- `Q`: sair salvando.

## Critérios

- `VISIBLE`: bola localizável com clareza;
- `PARTIAL`: bola parcialmente ocluída, borrada ou cortada, mas ainda localizável;
- `NOT_VISIBLE`: bola fora do frame ou sem evidência visual suficiente;
- `UNCERTAIN`: existe uma região suspeita, mas não é possível confirmar com segurança.

Não classificar `NOT_VISIBLE` como falso negativo: a bola não está disponível visualmente para o detector.

O lote de `train` será usado para treinamento. O lote de `validation` deverá permanecer separado para validação e ajuste controlado.

Ao executar novamente com `--pending-only`, o aplicativo carrega e preserva automaticamente o arquivo indicado em `--output`, avançando somente pelas amostras ainda pendentes.
