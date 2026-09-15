param(
    [switch]$ConfirmarExecucao
)

$ErrorActionPreference = 'Stop'
$raiz = 'C:\tcc_cv'
$executor = Join-Path $raiz 'scripts\run_goal_event_validation_miner_rocm_v1.ps1'
$saidaRelativa = 'experiments\ball\GOAL-EVENT-VALIDATION-MINER-V1\gpu_rocm_remaining_pairs_v1'
$saida = Join-Path $raiz $saidaRelativa
$lotes = @(
    @('match_05', 'match_12'),
    @('match_14', 'match_26'),
    @('match_30', 'match_33'),
    @('match_42')
)

if (-not (Test-Path -LiteralPath $executor)) {
    throw "Executor ROCm não encontrado: $executor"
}
if (Test-Path -LiteralPath $saida) {
    throw "A saída já existe; não é seguro sobrescrevê-la: $saida"
}

if (-not $ConfirmarExecucao) {
    Write-Output 'Pré-validação concluída. Use -ConfirmarExecucao para executar 7 partidas VALIDATION em quatro lotes sequenciais.'
    exit 0
}

New-Item -ItemType Directory -Force -Path $saida | Out-Null
$statusPath = Join-Path $saida 'status.json'
$inicio = Get-Date
$concluidas = @()

foreach ($indice in 0..($lotes.Count - 1)) {
    $lote = $lotes[$indice]
    @{
        status = 'running'
        started_at = $inicio.ToString('o')
        current_batch = $indice + 1
        current_matches = $lote
        completed_matches = $concluidas
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $statusPath

    $processos = foreach ($partida in $lote) {
        $saidaPartida = "$saidaRelativa\\$partida"
        $stdout = Join-Path $saida "$partida.stdout.log"
        $stderr = Join-Path $saida "$partida.stderr.log"
        $argumentos = @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $executor,
            '-Partidas', $partida,
            '-Saida', $saidaPartida,
            '-ConfirmarExecucao'
        )
        Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentos -WorkingDirectory $raiz -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    }

    Wait-Process -Id $processos.Id

    $falhas = @()
    foreach ($partida in $lote) {
        $progresso = Join-Path $saida "$partida\\progress.json"
        if (-not (Test-Path -LiteralPath $progresso)) {
            $falhas += "${partida}: progress.json ausente"
            continue
        }
        $estado = Get-Content -Raw -LiteralPath $progresso | ConvertFrom-Json
        if ($estado.status -ne 'completed') {
            $falhas += "${partida}: status $($estado.status)"
            continue
        }
        $concluidas += $partida
    }
    if ($falhas.Count -gt 0) {
        @{ status = 'failed'; started_at = $inicio.ToString('o'); failed_at = (Get-Date).ToString('o'); current_batch = $indice + 1; completed_matches = $concluidas; failures = $falhas } |
            ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $statusPath
        throw "Lote $($indice + 1) falhou: $($falhas -join '; ')"
    }
}

$fim = Get-Date
@{
    status = 'completed'
    started_at = $inicio.ToString('o')
    finished_at = $fim.ToString('o')
    wall_seconds = [math]::Round(($fim - $inicio).TotalSeconds, 3)
    completed_matches = $concluidas
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $statusPath
