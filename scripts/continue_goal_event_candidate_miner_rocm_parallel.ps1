param(
    [string]$WorkerPidsCsv,
    [string]$MatchA,
    [string]$MatchB,
    [string]$BatchStatusPath
)

$ErrorActionPreference = 'Stop'
$workerPids = @($WorkerPidsCsv -split ',' | ForEach-Object { [int]$_ })
while (@($workerPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
    Start-Sleep -Seconds 5
}

$root = Split-Path -Parent $PSScriptRoot
$executor = Join-Path $PSScriptRoot 'run_goal_event_candidate_miner_rocm_parallel_benchmark.ps1'
@{ status = 'running_second_batch'; second_batch = @($MatchA, $MatchB); started_at = (Get-Date).ToString('o') } |
    ConvertTo-Json | Set-Content -Encoding utf8 $BatchStatusPath
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $executor -MatchA $MatchA -MatchB $MatchB -ConfirmarExecucao
if ($LASTEXITCODE -ne 0) { throw "A segunda rodada falhou com código $LASTEXITCODE." }
@{ status = 'completed'; finished_at = (Get-Date).ToString('o'); second_batch = @($MatchA, $MatchB) } |
    ConvertTo-Json | Set-Content -Encoding utf8 $BatchStatusPath
