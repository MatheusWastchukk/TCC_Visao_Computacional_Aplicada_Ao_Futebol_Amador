$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}
Set-Location $repoRoot

function Test-Window([string]$Pattern) {
    return [bool](Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -match $Pattern } | Select-Object -First 1)
}

function Start-Queue([string]$Pattern, [string]$Name, [string[]]$Arguments) {
    if (Test-Window $Pattern) {
        Write-Output "$Name already open; keeping the existing annotation session."
        return
    }
    $process = Start-Process -FilePath $python -WorkingDirectory $repoRoot -ArgumentList $Arguments -PassThru
    Write-Output "$Name opened (PID $($process.Id))."
}

Start-Queue 'hard negatives' 'Hard-negative batch 2' @(
    'app\annotate_pairwise_control_audit.py',
    '--manifest', 'experiments\ball\GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001\pairwise_control_audit_v2_batch2.csv',
    '--output', 'experiments\ball\GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001\pairwise_control_audit_v2_batch2_annotated.csv',
    '--window-title', 'HardnegBatch2'
)

Start-Queue 'Identidade temporal da bola' 'BALL tracker annotation' @(
    'experiments\ball\annotate_ball_identity_sequence.py',
    '--input', 'experiments\ball\ORACLE-TRACKER-RETEST-EXP-0001\frame_annotations_v1.csv',
    '--pending-only'
)

Start-Queue 'BALL miss cause audit|Auditoria visual de misses BALL' 'BALL miss-cause audit' @(
    'app\annotate_ball_positive_miss_causes.py',
    '--input', 'experiments\ball\BALL-POSITIVE-MISS-AUDIT-EXP-0002\positive_miss_mechanism_v2.csv',
    '--output', 'experiments\ball\BALL-POSITIVE-MISS-AUDIT-EXP-0002\positive_miss_mechanism_v2_annotated.csv',
    '--window-title', 'BallMissAudit'
)

$watcherName = 'run_checkpoint100_when_review_complete.ps1'
$watcherPath = Join-Path $repoRoot 'experiments\ball\GOAL-EVENT-RANKING-ERROR-DIAGNOSTIC-EXP-0001\run_checkpoint100_when_review_complete.ps1'
$existingWatcher = @()
try {
    $existingWatcher = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction Stop |
        Where-Object { $_.CommandLine -like "*$watcherName*" })
}
catch {
    Write-Warning "Could not inspect existing watcher processes: $($_.Exception.Message)"
}
if ($existingWatcher.Count -gt 0) {
    Write-Output "Checkpoint100 watcher already running (PID $($existingWatcher[0].ProcessId))."
}
else {
    $watcherArgs = '-NoProfile -ExecutionPolicy Bypass -File "' + $watcherPath + '"'
    $watcher = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -WorkingDirectory $repoRoot -ArgumentList $watcherArgs -PassThru
    Write-Output "Checkpoint100 watcher started in background (PID $($watcher.Id))."
}

Write-Output 'The three OpenCV windows are visible, independent and resumable. Use Alt+Tab to switch between them.'
