param(
    [string]$MatchA = 'match_21',
    [string]$MatchB = 'match_22',
    [switch]$ConfirmarExecucao
)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent $PSScriptRoot
$manifesto = Join-Path $raiz 'experiments\datasets\dataset_split_v1.csv'
$videos = Join-Path $raiz 'data\Videos_Edited'
$python = Join-Path $raiz '.venv-rocm\Scripts\python.exe'
$minerador = Join-Path $raiz 'experiments\ball\mine_goal_event_candidates_v1.py'
$modeloGol = Join-Path $raiz 'runs\detect\experiments\goal_geometry\EXP-0092\yolo11n_cpu_reference\weights\best.pt'
$modeloBola = Join-Path $raiz 'runs\detect\experiments\ball\BALL-EXP-0005\weights\best.pt'
$saidaRelativa = "experiments\ball\GOAL-EVENT-CANDIDATE-MINER-V1\gpu_rocm_parallel_full_$($MatchA -replace '_','')_$($MatchB -replace '_','')_v1"
$saida = Join-Path $raiz $saidaRelativa

foreach ($arquivo in @($manifesto, $python, $minerador, $modeloGol, $modeloBola)) {
    if (-not (Test-Path -LiteralPath $arquivo)) { throw "Arquivo obrigatório ausente: $arquivo" }
}
if ($MatchA -eq $MatchB) { throw 'As duas partidas precisam ser diferentes.' }
if (Test-Path -LiteralPath $saida) { throw "A saída já existe: $saida" }

$partidas = Import-Csv $manifesto | Where-Object { $_.match_id -in @($MatchA, $MatchB) }
if ($partidas.Count -ne 2) { throw 'Partida ausente no manifesto.' }
foreach ($partida in $partidas) {
    if ($partida.split -ne 'train') { throw "$($partida.match_id) não pertence a TRAIN." }
    $video = Join-Path $videos $partida.video_filename
    if (-not (Test-Path -LiteralPath $video)) { throw "Vídeo ausente: $video" }
}

if (-not $ConfirmarExecucao) {
    Write-Output "Pré-validação concluída para $MatchA e $MatchB. Use -ConfirmarExecucao para iniciar o benchmark."
    exit 0
}

New-Item -ItemType Directory -Force $saida | Out-Null
$cache = Join-Path $raiz '.rocm_cache\miopen_cache'
$banco = Join-Path $raiz '.rocm_cache\miopen_user_db'
New-Item -ItemType Directory -Force $cache, $banco | Out-Null
$env:MIOPEN_CUSTOM_CACHE_DIR = $cache
$env:MIOPEN_USER_DB_PATH = $banco

$amostras = Join-Path $saida 'usage_samples.csv'
$estado = Join-Path $saida 'status.json'
$inicio = Get-Date
@{ status = 'running'; started_at = $inicio.ToString('o'); matches = @($MatchA, $MatchB) } |
    ConvertTo-Json | Set-Content -Encoding utf8 $estado

function Argumentos-Minerador([string]$match, [string]$destino) {
    return @(
        'experiments\ball\mine_goal_event_candidates_v1.py', '--split-manifest', 'experiments\datasets\dataset_split_v1.csv', '--videos-dir', 'data\Videos_Edited',
        '--goal-model', 'runs\detect\experiments\goal_geometry\EXP-0092\yolo11n_cpu_reference\weights\best.pt', '--ball-model', 'runs\detect\experiments\ball\BALL-EXP-0005\weights\best.pt',
        '--include-matches', $match, '--output', $destino, '--device', '0'
    )
}

$saidasPartidas = @{}
foreach ($partida in $partidas) {
    $destino = Join-Path $saidaRelativa $partida.match_id
    $saidasPartidas[$partida.match_id] = $destino
}
$processos = @(
    Start-Process -FilePath $python -ArgumentList (Argumentos-Minerador $MatchA $saidasPartidas[$MatchA]) -WorkingDirectory $raiz -WindowStyle Hidden -PassThru
    Start-Process -FilePath $python -ArgumentList (Argumentos-Minerador $MatchB $saidasPartidas[$MatchB]) -WorkingDirectory $raiz -WindowStyle Hidden -PassThru
)

$contadorDisponivel = $true
$linhas = @()
while ($processos | Where-Object { -not $_.HasExited }) {
    $agora = Get-Date
    $porPid = @{}
    $trabalhadores = Get-Process python, python3 -ErrorAction SilentlyContinue |
        Where-Object { $_.StartTime -ge $inicio.AddSeconds(-2) -and $_.WorkingSet64 -gt 500MB }
    try {
        $medidas = (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples
        foreach ($trabalhador in $trabalhadores) {
            $padrao = "pid_$($trabalhador.Id)_"
            $porPid[$trabalhador.Id] = [math]::Round((@($medidas | Where-Object { $_.InstanceName -like "*$padrao*" } | Measure-Object -Property CookedValue -Sum).Sum), 3)
        }
    } catch {
        $contadorDisponivel = $false
    }
    foreach ($trabalhador in $trabalhadores) {
        $atual = Get-Process -Id $trabalhador.Id -ErrorAction SilentlyContinue
        if ($null -eq $atual) { continue }
        $linha = [PSCustomObject]@{
            timestamp = $agora.ToString('o')
            pid = $trabalhador.Id
            cpu_seconds = [math]::Round($atual.CPU, 3)
            working_set_mb = [math]::Round($atual.WorkingSet64 / 1MB, 2)
            gpu_engine_utilization_sum_percent = if ($porPid.ContainsKey($trabalhador.Id)) { $porPid[$trabalhador.Id] } else { $null }
        }
        $linhas += $linha
        $linha | Export-Csv -NoTypeInformation -Append -Encoding utf8 $amostras
    }
    Start-Sleep -Seconds 5
}
foreach ($processo in $processos) { $processo.WaitForExit() }
$fim = Get-Date
$codigos = @{}
foreach ($processo in $processos) { $codigos[[string]$processo.Id] = $processo.ExitCode }
$gpu = @($linhas | Where-Object { $null -ne $_.gpu_engine_utilization_sum_percent } | Select-Object -ExpandProperty gpu_engine_utilization_sum_percent)
$resumo = [ordered]@{
    experiment_id = Split-Path -Leaf $saida
    status = if (($codigos.Values | Where-Object { $_ -ne 0 }).Count -eq 0) { 'completed' } else { 'failed' }
    scope = 'somente TRAIN'
    started_at = $inicio.ToString('o')
    finished_at = $fim.ToString('o')
    wall_seconds = [math]::Round(($fim - $inicio).TotalSeconds, 3)
    matches = @($MatchA, $MatchB)
    worker_exit_codes = $codigos
    usage_samples = $linhas.Count
    gpu_counter_available = $contadorDisponivel -and $gpu.Count -gt 0
    gpu_engine_utilization_sum_mean_percent = if ($gpu.Count) { [math]::Round((($gpu | Measure-Object -Average).Average), 3) } else { $null }
    gpu_engine_utilization_sum_max_percent = if ($gpu.Count) { [math]::Round((($gpu | Measure-Object -Maximum).Maximum), 3) } else { $null }
    note = 'Benchmark de infraestrutura; a soma dos motores GPU pode exceder 100 e não representa avaliação de eventos.'
}
$resumo | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $saida 'metrics.json')
$resumo | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $estado
Write-Output ($resumo | ConvertTo-Json -Depth 4)
