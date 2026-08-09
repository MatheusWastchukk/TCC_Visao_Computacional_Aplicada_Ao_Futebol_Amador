# Goal Detection Pilot

Piloto para decidir se a localizacao classica da estrutura do gol e viavel antes da migracao para partidas completas.

## Fluxo

1. Catalogar 3 partidas em `matches.csv`.
2. Selecionar timestamps positivos e negativos.
3. Extrair frames preservando resolucao e origem.
4. Anotar `main_goal`, `secondary_goal` e `not_goal`.
5. Executar configuracoes de limiarizacao, morfologia, bordas, linhas e geometria.
6. Registrar cada tentativa em `experiment_matrix.csv`.
7. Avaliar frame a frame, temporalmente e em trechos continuos.
8. Escrever `REPORT.md` com decisao `GO`, `GO_HIBRIDO` ou `NO_GO`.

## O que versionar

Versionar arquivos pequenos:

- configuracoes;
- CSVs de metadados e matriz;
- anotacoes leves;
- relatorios;
- scripts.

Nao versionar videos, frames extraidos, mascaras, bordas, imagens de linhas, clipes ou outros artefatos gerados.

## Extracao de frames

Modelo de entrada: `timestamps_template.csv`.
Guia de preenchimento: `TEMPLATE_GUIDE.md`.

Exemplo:

```powershell
python experiments/goal_detection_pilot/scripts/extract_frames.py `
  --timestamps experiments/goal_detection_pilot/timestamps_template.csv `
  --output-dir experiments/goal_detection_pilot/data/frames
```

## Baseline classico

Depois de extrair os frames, execute as 12 combinacoes de limiarizacao e
morfologia:

```powershell
python experiments/goal_detection_pilot/scripts/run_baseline.py `
  --timestamps experiments/goal_detection_pilot/timestamps_template.csv `
  --frames-dir experiments/goal_detection_pilot/data/frames `
  --config experiments/goal_detection_pilot/configs/pilot_config.yml `
  --matrix-output experiments/goal_detection_pilot/experiment_matrix.csv `
  --report-output experiments/goal_detection_pilot/outputs/reports/baseline_report.json
```

O baseline classifica `main_goal` contra `not_goal` e `secondary_goal`. IoU,
falsos candidatos por hora e estabilidade temporal exigem, respectivamente,
caixas anotadas e trechos continuos; nao sao inferidos a partir desta amostra
de timestamps isolados.
