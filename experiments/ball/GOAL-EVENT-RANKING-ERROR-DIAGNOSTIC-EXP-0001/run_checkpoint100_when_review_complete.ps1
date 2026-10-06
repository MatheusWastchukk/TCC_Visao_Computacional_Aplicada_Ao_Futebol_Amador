$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$annotated = Join-Path $PSScriptRoot 'pairwise_control_audit_v2_batch2_annotated.csv'
$prior = Join-Path $PSScriptRoot 'pairwise_control_audit_v2_prior50_annotated.csv'
$output = Join-Path $PSScriptRoot 'PROGRESSIVE-HARDNEG-EXP-0001-CHECKPOINT100'
$log = Join-Path $PSScriptRoot 'checkpoint100_watcher.log'
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $PSScriptRoot 'evaluate_progressive_hardneg_lomo_v1.py'

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

Write-Log 'Watcher started; waiting for all 50 batch-2 candidates to be manually completed.'
if (-not (Test-Path -LiteralPath $prior)) { throw "Prior confirmed controls missing: $prior" }
if (Test-Path -LiteralPath $output) { throw "Output already exists; refusing overwrite: $output" }

$priorRows = Import-Csv -LiteralPath $prior
if ($priorRows.Count -ne 50) { throw "Expected 50 prior controls; got $($priorRows.Count)" }
if (@($priorRows | Where-Object { $_.manual_status -ne 'COMPLETED' -or $_.manual_label -ne 'NO_GOAL' }).Count -ne 0) {
    throw 'Prior controls contain an unconfirmed or non-NO_GOAL row.'
}
$priorIds = @($priorRows | ForEach-Object { $_.control_group_uid })
if (@($priorIds | Select-Object -Unique).Count -ne $priorIds.Count) { throw 'Prior controls contain duplicate IDs.' }

$lastState = ''
while (-not (Test-Path -LiteralPath $annotated)) { Start-Sleep -Seconds 20 }

while ($true) {
    try {
        $rows = @(Import-Csv -LiteralPath $annotated)
        $completed = @($rows | Where-Object { $_.manual_status -eq 'COMPLETED' })
        $pending = $rows.Count - $completed.Count
        $state = "rows=$($rows.Count);completed=$($completed.Count);pending=$pending"
        if ($state -ne $lastState) { Write-Log "Review progress: $state"; $lastState = $state }

        if ($rows.Count -eq 50 -and $completed.Count -eq 50) {
            $badLabels = @($rows | Where-Object { $_.manual_label -notin @('NO_GOAL', 'GOAL', 'NOT_EVALUABLE') })
            if ($badLabels.Count -gt 0) { throw "Completed batch contains $($badLabels.Count) invalid labels." }
            $newIds = @($rows | ForEach-Object { $_.control_group_uid })
            if (@($newIds | Select-Object -Unique).Count -ne $newIds.Count) { throw 'Batch 2 contains duplicate IDs.' }
            if (@($newIds | Where-Object { $_ -in $priorIds }).Count -gt 0) { throw 'Batch 2 overlaps prior controls.' }
            if (Test-Path -LiteralPath $output) { throw "Output already exists; refusing overwrite: $output" }

            Write-Log 'All batch-2 reviews completed; running frozen checkpoint100 evaluator.'
            Push-Location $repo
            try {
                & $python $runner --controls $annotated --prior-controls $prior --output $output
                $exitCode = $LASTEXITCODE
            }
            finally { Pop-Location }
            Write-Log "Evaluator exited with code $exitCode."
            exit $exitCode
        }
    }
    catch {
        Write-Log "Check failed; waiting for a stable saved CSV: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds 20
}
