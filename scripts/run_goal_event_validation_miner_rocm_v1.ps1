param(
    [Parameter(Mandatory = $true)]
    [string[]]$Partidas,

    [Parameter(Mandatory = $true)]
    [string]$Saida,

    [switch]$ConfirmarExecucao
)

$ErrorActionPreference = 'Stop'

# A junção curta evita a falha de aspas do offload-arch quando o caminho físico
# do repositório contém espaço. Ela é uma pré-condição conhecida deste ambiente.
$raiz = 'C:\tcc_cv'
$python = Join-Path $raiz '.venv-rocm\Scripts\python.exe'
$minerador = Join-Path $raiz 'experiments\ball\mine_goal_event_candidates_v1.py'
$saidaAbsoluta = Join-Path $raiz $Saida

if (-not (Test-Path -LiteralPath $python)) {
    throw "Ambiente ROCm não encontrado em: $python"
}
if (-not (Test-Path -LiteralPath $minerador)) {
    throw "Minerador não encontrado em: $minerador"
}
if (Test-Path -LiteralPath $saidaAbsoluta) {
    throw "A saída já existe; escolha um diretório novo: $saidaAbsoluta"
}

if (-not $ConfirmarExecucao) {
    Write-Output 'Pré-validação concluída. Use -ConfirmarExecucao para iniciar a mineração VALIDATION na GPU.'
    exit 0
}

# Cada execução usa banco e cache próprios, evitando locks compartilhados do MIOpen.
$identificador = ($Partidas -join '_').Replace('match_', 'm')
$cacheRaiz = Join-Path $raiz ".rocm-cache\$identificador"
$bancoMIOpen = Join-Path $cacheRaiz 'miopen-db'
$cacheMIOpen = Join-Path $cacheRaiz 'miopen-kernels'
New-Item -ItemType Directory -Force -Path $bancoMIOpen, $cacheMIOpen | Out-Null

$env:PYTORCH_ROCM_ARCH = 'gfx1200'
$env:MIOPEN_USER_DB_PATH = $bancoMIOpen
$env:MIOPEN_CUSTOM_CACHE_DIR = $cacheMIOpen

& $python $minerador `
    --split-manifest 'experiments\datasets\dataset_split_v1.csv' `
    --videos-dir 'data\Videos_Edited' `
    --goal-model 'runs\detect\experiments\goal_geometry\EXP-0092\yolo11n_cpu_reference\weights\best.pt' `
    --ball-model 'runs\detect\experiments\ball\BALL-EXP-0005\weights\best.pt' `
    --include-matches $Partidas `
    --expected-split validation `
    --output $Saida `
    --device 0

if ($LASTEXITCODE -ne 0) {
    throw "A mineração falhou com código $LASTEXITCODE. Consulte stderr e preserve a saída para diagnóstico."
}
