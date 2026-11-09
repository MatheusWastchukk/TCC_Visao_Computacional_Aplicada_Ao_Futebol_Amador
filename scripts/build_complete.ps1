param([switch]$RebuildEngines)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falha ao executar $Program (código $LASTEXITCODE)." }
}
Push-Location $projectRoot
try {
    if ($RebuildEngines) {
        Invoke-Checked 'powershell.exe' @('-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/build_cpu.ps1','-BackendOnly')
        foreach ($mode in @('nvidia','amd')) {
            Invoke-Checked 'powershell.exe' @('-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/build_accelerator.ps1','-Mode',$mode)
        }
    }
    Push-Location desktop
    try {
        Invoke-Checked 'npm.cmd' @('run','build')
        Invoke-Checked 'npm.cmd' @('run','pack:complete')
    } finally { Pop-Location }
    $package = Get-Content -Raw -Encoding UTF8 desktop/package.json | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath 'build/tools/7zip/7zr.exe')) {
        Invoke-Checked '.venv-cpu/Scripts/python.exe' @('scripts/fetch_packaging_tools.py')
    }
    Invoke-Checked '.venv-cpu/Scripts/python.exe' @('scripts/package_complete.py','--source','dist/complete/win-unpacked',
      '--output',"dist/complete/GoalEventReview-$($package.version)-offline-x64.exe")
    Write-Host 'Pacote completo offline gerado. Os três componentes serão preparados apenas na primeira abertura.'
} finally { Pop-Location }
