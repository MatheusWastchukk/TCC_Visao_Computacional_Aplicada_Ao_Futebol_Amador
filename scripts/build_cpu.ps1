param(
    [switch]$Setup,
    [switch]$Installer,
    [switch]$BackendOnly,
    [string]$Python = "python",
    [string]$EncoderWeights = "$env:USERPROFILE\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$cpuPython = Join-Path $projectRoot ".venv-cpu\Scripts\python.exe"

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falha ao executar $Program (código $LASTEXITCODE)." }
}

Push-Location $projectRoot
try {
    if ($Setup) {
        if (-not (Test-Path -LiteralPath $cpuPython)) {
            Invoke-Checked $Python @("-m", "venv", ".venv-cpu")
        }
        Invoke-Checked $cpuPython @("-m", "pip", "install", "torch==2.9.1+cpu", "torchvision==0.24.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu")
        Invoke-Checked $cpuPython @("-m", "pip", "install", "-r", "requirements-cpu.txt", "-r", "requirements-build-cpu.txt")
        Push-Location desktop
        try {
            Invoke-Checked "npm.cmd" @("ci", "--ignore-scripts")
            # npm recente pode deixar scripts de dependências pendentes de aprovação.
            # Estes dois scripts obtêm apenas os runtimes usados pelo aplicativo.
            Invoke-Checked "node" @("node_modules/electron/install.js")
            Invoke-Checked "node" @("node_modules/ffmpeg-static/install.js")
        } finally { Pop-Location }
    }
    if (-not (Test-Path -LiteralPath $cpuPython)) { throw "Prepare o ambiente com -Setup e Python 3.12." }
    if (-not (Test-Path -LiteralPath $EncoderWeights)) { throw "Informe -EncoderWeights com os pesos resnet18-f37072fd.pth existentes. O build não baixa modelos automaticamente." }
    Invoke-Checked $cpuPython @("-c", "import sys, torch; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12'; assert torch.version.cuda is None and getattr(torch.version, 'hip', None) is None, 'Use PyTorch CPU'")
    $env:GOAL_REVIEW_BUILD_RESNET = (Resolve-Path -LiteralPath $EncoderWeights).Path
    $env:GOAL_REVIEW_BUILD_RUNTIME = 'cpu'
    $env:GOAL_REVIEW_BUILD_FOLDER = 'goal-review-api'
    Invoke-Checked $cpuPython @("-m", "pip", "check")
    Invoke-Checked $cpuPython @("-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", "build/cpu/backend", "--workpath", "build/cpu/pyinstaller", "packaging/backend_cpu.spec")
    $previousDevice = $env:GOAL_REVIEW_DEVICE
    try {
        $env:GOAL_REVIEW_DEVICE = "cpu"
        Invoke-Checked (Join-Path $projectRoot "build/cpu/backend/goal-review-api/goal-review-api.exe") @("--self-check")
    } finally { $env:GOAL_REVIEW_DEVICE = $previousDevice }
    $package = Get-Content -Raw -Encoding UTF8 desktop/package.json | ConvertFrom-Json
    @{mode='cpu';app_version=$package.version} | ConvertTo-Json | Set-Content -Encoding UTF8 'build/cpu/backend/goal-review-api/component.json'
    if ($BackendOnly) { Write-Host 'Backend CPU construído e verificado.'; return }
    Push-Location desktop
    try {
        Invoke-Checked "npm.cmd" @("run", "build")
        if ($Installer) { Invoke-Checked "npm.cmd" @("run", "dist:cpu") }
        else { Invoke-Checked "npm.cmd" @("run", "pack:cpu") }
    } finally { Pop-Location }
    Invoke-Checked $cpuPython @("scripts/write_cpu_build_manifest.py", "--package", "dist/cpu/win-unpacked", "--output", "build/cpu/build_manifest.json")
    Write-Host "Pacote preliminar gerado em dist/cpu. Ainda exige teste em outro computador."
} finally {
    Pop-Location
}
