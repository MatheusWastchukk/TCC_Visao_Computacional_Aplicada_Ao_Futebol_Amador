param(
    [Parameter(Mandatory=$true)][ValidateSet('nvidia','amd')][string]$Mode,
    [switch]$Setup,
    [switch]$Zip,
    [string]$Python,
    [string]$EncoderWeights = "$env:USERPROFILE\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth"
)
$ErrorActionPreference = 'Stop'
$Mode = $Mode.ToLowerInvariant()
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $environment = if ($Mode -eq 'nvidia') { '.venv-cuda' } else { '.venv-rocm' }
    $Python = Join-Path $projectRoot "$environment\Scripts\python.exe"
}
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falha ao executar $Program (código $LASTEXITCODE)." }
}
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Prepare primeiro um ambiente Python 3.12 isolado com PyTorch CUDA ou ROCm.' }
    if ($Setup) {
        if ($Mode -eq 'nvidia') {
            Invoke-Checked $Python @('-m','pip','install','torch==2.9.1+cu126','torchvision==0.24.1+cu126','--index-url','https://download.pytorch.org/whl/cu126')
        }
        Invoke-Checked $Python @('-m','pip','install','-r','requirements-gpu-runtime.txt','-r','requirements-build-cpu.txt')
    }
    $env:GOAL_REVIEW_BUILD_RUNTIME = $Mode
    $env:GOAL_REVIEW_BUILD_RESNET = (Resolve-Path -LiteralPath $EncoderWeights).Path
    $env:GOAL_REVIEW_DEVICE = 'cpu'
    $env:GOAL_REVIEW_ACCELERATOR = 'cpu'
    $distRoot = "build/engines/$Mode"
    $env:GOAL_REVIEW_BUILD_FOLDER = 'goal-review-api'
    if ($Mode -eq 'amd') {
        # Os nomes dos kernels ROCm podem ultrapassar MAX_PATH no checkout longo.
        $distRoot = Join-Path ([System.IO.Path]::GetTempPath()) 'ger42a'
        $env:GOAL_REVIEW_BUILD_FOLDER = 'r'
    }
    Invoke-Checked $Python @('-m','pip','check')
    Invoke-Checked $Python @('-m','PyInstaller','--noconfirm','--distpath',$distRoot,'--workpath',"build/engines-work/$Mode",'packaging/backend_cpu.spec')
    $component = Join-Path $distRoot $env:GOAL_REVIEW_BUILD_FOLDER
    $component = [System.IO.Path]::GetFullPath($component)
    Invoke-Checked (Join-Path $component 'goal-review-api.exe') @('--self-check','--expected-runtime',$Mode)
    $package = Get-Content -Raw -Encoding UTF8 desktop/package.json | ConvertFrom-Json
    $manifest = @{ mode=$Mode; app_version=$package.version; validation='Runtime e modelos verificados em CPU; a GPU é validada na seleção.' }
    $manifest | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $component 'component.json')
    if ($Mode -eq 'amd') {
        $target = Join-Path $projectRoot "build/engines/$Mode/goal-review-api"
        Invoke-Checked $Python @('scripts/copy_component.py','--source',$component,'--target',$target)
        $component = $target
    }
    if ($Zip) {
        Invoke-Checked $Python @('scripts/package_accelerator.py','--mode',$Mode,'--source',$component,'--output',"dist/engines/GoalEventReview-$($package.version)-$Mode.zip")
    }
    Write-Host "Componente $Mode construído em $component. A disponibilidade da GPU será validada pelo aplicativo."
} finally { Pop-Location }
