param([switch]$Setup)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falha ao executar $Program (código $LASTEXITCODE)." }
}
Push-Location $projectRoot
try {
    if ($Setup -and -not (Test-Path -LiteralPath 'build/tools/inno7/ISCC.exe')) {
        Invoke-Checked '.venv-cpu/Scripts/python.exe' @('scripts/run_guarded.py','--min-free-gb','3',
          '--report','build/guard/inno_compiler_setup.json','--','.venv-cpu/Scripts/python.exe','scripts/fetch_installer_compiler.py')
    }
    if (-not (Test-Path -LiteralPath 'build/tools/inno7/ISCC.exe')) {
        throw 'Prepare o compilador com -Setup.'
    }
    Invoke-Checked '.venv-cpu/Scripts/python.exe' @('scripts/run_guarded.py','--min-free-gb','3',
      '--report','build/guard/installer_build.json','--','.venv-cpu/Scripts/python.exe','scripts/build_installer.py')
    Write-Host 'Instalador completo gerado em dist/installer. Envie o arquivo Setup-x64.exe para teste.'
} finally { Pop-Location }
