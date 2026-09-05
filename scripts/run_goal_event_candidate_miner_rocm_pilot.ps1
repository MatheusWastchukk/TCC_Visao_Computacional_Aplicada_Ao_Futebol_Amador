<#!
.SYNOPSIS
Prepara e, somente com confirmação explícita, executa o piloto ROCm do minerador de eventos.

.DESCRIPTION
O script valida pré-requisitos sem carregar modelos. A mineração só começa com
-ConfirmarExecucao, para evitar execução acidental enquanto outro experimento estiver ativo.
#>
[CmdletBinding()]
param(
    [switch]$ConfirmarExecucao
)

$ErrorActionPreference = 'Stop'
$raizProjeto = Split-Path -Parent $PSScriptRoot
$pythonRocm = Join-Path $raizProjeto '.venv-rocm\Scripts\python.exe'
$manifestoSplit = Join-Path $raizProjeto 'experiments\datasets\dataset_split_v1.csv'
$videosDir = Join-Path $raizProjeto 'data\Videos_Edited'
$modeloGol = Join-Path $raizProjeto 'runs\detect\experiments\goal_geometry\EXP-0092\yolo11n_cpu_reference\weights\best.pt'
$modeloBola = Join-Path $raizProjeto 'runs\detect\experiments\ball\BALL-EXP-0005\weights\best.pt'
$saida = Join-Path $raizProjeto 'experiments\ball\GOAL-EVENT-CANDIDATE-MINER-V1\pilot_train_3_matches_rocm'
$partidas = @('match_02', 'match_04', 'match_06')

foreach ($caminho in @($pythonRocm, $manifestoSplit, $videosDir, $modeloGol, $modeloBola)) {
    if (-not (Test-Path -LiteralPath $caminho)) {
        throw "Pré-requisito ausente: $caminho"
    }
}
if (Test-Path -LiteralPath $saida) {
    throw "A saída já existe: $saida. Escolha uma pasta nova para não sobrescrever evidências."
}

$splits = Import-Csv -LiteralPath $manifestoSplit
foreach ($partida in $partidas) {
    $registro = $splits | Where-Object { $_.match_id -eq $partida }
    if ($null -eq $registro -or $registro.split -ne 'train') {
        throw "A partida $partida não pertence a TRAIN; o piloto ROCm foi bloqueado."
    }
}

Write-Host 'Pré-requisitos ROCm validados. Nenhum modelo foi carregado.'
Write-Host "Partidas autorizadas: $($partidas -join ', ')"
Write-Host "Saída planejada: $saida"
if (-not $ConfirmarExecucao) {
    Write-Host 'Execução não iniciada. Use -ConfirmarExecucao somente após autorização explícita.'
    exit 0
}

$cacheDb = Join-Path $raizProjeto '.rocm_cache\miopen_db'
$cacheKernel = Join-Path $raizProjeto '.rocm_cache\miopen_kernel'
New-Item -ItemType Directory -Force -Path $cacheDb, $cacheKernel | Out-Null
$env:MIOPEN_USER_DB_PATH = $cacheDb
$env:MIOPEN_SYSTEM_DB_PATH = $cacheDb
$env:MIOPEN_CUSTOM_CACHE_DIR = $cacheKernel

& $pythonRocm (Join-Path $raizProjeto 'experiments\ball\mine_goal_event_candidates_v1.py') `
    --split-manifest $manifestoSplit `
    --videos-dir $videosDir `
    --goal-model $modeloGol `
    --ball-model $modeloBola `
    --include-matches $partidas `
    --output $saida `
    --device 0
