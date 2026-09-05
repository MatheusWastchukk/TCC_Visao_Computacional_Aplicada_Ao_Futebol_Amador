param(
    [switch]$ConfirmarExecucao
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv-rocm\Scripts\python.exe'
$miner = Join-Path $root 'experiments\ball\mine_goal_event_candidates_v1.py'
$base = Join-Path $root 'experiments\ball\GOAL-EVENT-CANDIDATE-MINER-V1'
$statusDirectory = Join-Path $base 'gpu_rocm_final_sequential_status_v1'
$stages = @(
    @{ matches = @('match_35', 'match_36', 'match_37'); output = 'experiments\ball\GOAL-EVENT-CANDIDATE-MINER-V1\gpu_rocm_sequential_match35_match37_v1' },
    @{ matches = @('match_38', 'match_41'); output = 'experiments\ball\GOAL-EVENT-CANDIDATE-MINER-V1\gpu_rocm_sequential_match38_match41_v1' }
)

foreach ($stage in $stages) {
    if (Test-Path (Join-Path $root $stage.output)) { throw "A saída já existe: $($stage.output)" }
}
if (-not $ConfirmarExecucao) {
    Write-Output 'Pré-validação concluída. Use -ConfirmarExecucao para minerar as cinco partidas TRAIN em duas etapas sequenciais.'
    exit 0
}

New-Item -ItemType Directory -Force $statusDirectory | Out-Null
$cache = Join-Path $root '.rocm_cache\miopen_cache'
$database = Join-Path $root '.rocm_cache\miopen_user_db'
New-Item -ItemType Directory -Force $cache, $database | Out-Null
$env:MIOPEN_CUSTOM_CACHE_DIR = $cache
$env:MIOPEN_USER_DB_PATH = $database
$statusPath = Join-Path $statusDirectory 'status.json'
$started = Get-Date
@{ status = 'running_stage_1'; started_at = $started.ToString('o'); stages = $stages.matches } |
    ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $statusPath

foreach ($index in 0..($stages.Count - 1)) {
    $stage = $stages[$index]
    @{ status = "running_stage_$($index + 1)"; matches = $stage.matches; started_at = (Get-Date).ToString('o') } |
        ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $statusPath
    & $python $miner --split-manifest 'experiments\datasets\dataset_split_v1.csv' --videos-dir 'data\Videos_Edited' --goal-model 'runs\detect\experiments\goal_geometry\EXP-0092\yolo11n_cpu_reference\weights\best.pt' --ball-model 'runs\detect\experiments\ball\BALL-EXP-0005\weights\best.pt' --include-matches $stage.matches --output $stage.output --device 0
    if ($LASTEXITCODE -ne 0) { throw "A etapa $($index + 1) falhou com código $LASTEXITCODE." }
}

$finished = Get-Date
@{ status = 'completed'; started_at = $started.ToString('o'); finished_at = $finished.ToString('o'); wall_seconds = [math]::Round(($finished - $started).TotalSeconds, 3) } |
    ConvertTo-Json | Set-Content -Encoding utf8 $statusPath
