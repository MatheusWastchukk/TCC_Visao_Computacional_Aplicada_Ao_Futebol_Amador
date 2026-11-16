param([switch]$RebuildBase, [switch]$Setup)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-cpu/Scripts/python.exe'
function Invoke-Guarded {
    param([string]$Report, [string[]]$Command)
    & $python scripts/run_guarded.py --min-free-gb 3 --report $Report -- @Command
    if ($LASTEXITCODE -ne 0) { throw 'A tarefa falhou ou foi interrompida pelo supervisor. Consulte o relatório.' }
}
Push-Location $projectRoot
try {
    if ($RebuildBase) {
        $arguments = @('powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/build_cpu.ps1')
        if ($Setup) { $arguments += '-Setup' }
        Invoke-Guarded 'build/guard/online_base.json' $arguments
    }
    if (-not (Test-Path -LiteralPath 'build/tools/inno7/ISCC.exe')) {
        Invoke-Guarded 'build/guard/online_compiler.json' @($python,'scripts/fetch_installer_compiler.py')
    }
    Invoke-Guarded 'build/guard/online_installer.json' @($python,'scripts/build_online_installer.py')
} finally { Pop-Location }
