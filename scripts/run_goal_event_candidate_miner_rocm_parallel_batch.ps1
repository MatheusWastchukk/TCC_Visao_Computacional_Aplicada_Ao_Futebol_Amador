param(
    [switch]$ConfirmarExecucao
)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent $PSScriptRoot
$executor = Join-Path $PSScriptRoot 'run_goal_event_candidate_miner_rocm_parallel_benchmark.ps1'
$saida = Join-Path $raiz 'experiments\ball\GOAL-EVENT-CANDIDATE-MINER-V1\gpu_rocm_parallel_batch_match25_match32_v1'
if (Test-Path -LiteralPath $saida) { throw "A saída já existe: $saida" }

if (-not $ConfirmarExecucao) {
    Write-Output 'Pré-validação concluída. Use -ConfirmarExecucao para executar duas rodadas sequenciais de duas partidas paralelas.'
    exit 0
}

New-Item -ItemType Directory -Force $saida | Out-Null
$statusPath = Join-Path $saida 'status.json'
$started = Get-Date
@{ status = 'running'; started_at = $started.ToString('o'); batches = @(@('match_25', 'match_27'), @('match_28', 'match_32')) } |
    ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $statusPath

foreach ($batch in @(@('match_25', 'match_27'), @('match_28', 'match_32'))) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $executor -MatchA $batch[0] -MatchB $batch[1] -ConfirmarExecucao
    if ($LASTEXITCODE -ne 0) { throw "A rodada $($batch -join ', ') falhou com código $LASTEXITCODE." }
}

$finished = Get-Date
@{ status = 'completed'; started_at = $started.ToString('o'); finished_at = $finished.ToString('o'); wall_seconds = [math]::Round(($finished - $started).TotalSeconds, 3) } |
    ConvertTo-Json | Set-Content -Encoding utf8 $statusPath
