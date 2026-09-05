# GOAL-EVENT-CANDIDATE-MINER-V1

O minerador percorre exclusivamente os vídeos marcados como `train` no manifesto,
detecta estrutura do gol e bola no mesmo quadro, agrupa ocorrências próximas e gera
clipes candidatos para revisão humana.

Ele é um gerador de hipóteses, não um detector final de gol. Cada linha de
`candidate_events_v1.csv` deve ser revisada no anotador de clipes como `GOAL`,
`NO_GOAL` ou `NOT_EVALUABLE`.

## Proteções

- Recusa qualquer partida que não pertença a `train`.
- Recusa uma pasta de saída já preenchida.
- Não usa placar, reinício ou ResNet para eliminar candidatos.
- Não modifica os vídeos nem o manifesto de split.

## Saídas esperadas após execução autorizada

- `candidate_events_v1.csv`: fila de clipes pronta para revisão humana.
- `candidate_frames_v1.csv`: evidência por quadro que originou cada hipótese.
- `metrics.json`: parâmetros, contagens e resumo por partida.

## Comando preparado, mas não executar sem autorização

```powershell
.\.venv\Scripts\python.exe experiments\ball\mine_goal_event_candidates_v1.py `
  --split-manifest experiments\datasets\dataset_split_v1.csv `
  --videos-dir data\Videos_Edited `
  --goal-model runs\detect\experiments\goal_geometry\EXP-0092\yolo11n_cpu_reference\weights\best.pt `
  --ball-model runs\detect\experiments\ball\BALL-EXP-0005\weights\best.pt `
  --include-matches match_02 match_04 match_06 `
  --output experiments\ball\GOAL-EVENT-CANDIDATE-MINER-V1\pilot_train_3_matches
```

O primeiro ensaio recomendado é limitado a três partidas e usa CPU. Só depois da
revisão dos candidatos desse piloto deve-se ampliar para as demais partidas TRAIN.

## Preparação ROCm

O arquivo `scripts/run_goal_event_candidate_miner_rocm_pilot.ps1` valida o ambiente
ROCm, os pesos, as três partidas TRAIN e a inexistência da pasta de saída. Sem o
parâmetro `-ConfirmarExecucao`, ele não carrega modelos nem inicia mineração.

```powershell
.\scripts\run_goal_event_candidate_miner_rocm_pilot.ps1
```

Após uma autorização explícita para a execução ROCm, o mesmo comando deve receber
`-ConfirmarExecucao`. Esse piloto usará uma pasta de saída diferente da execução CPU.
