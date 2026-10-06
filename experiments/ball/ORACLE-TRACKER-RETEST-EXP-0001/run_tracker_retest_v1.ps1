$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repoRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$experiment = Join-Path $repoRoot 'experiments\ball\ORACLE-TRACKER-RETEST-EXP-0001'
$runner = 'experiments\ball\run_ball_multitrack_pilot.py'
$baseline = 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\baseline\weights\last.pt'
$oversampled = 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\oversampled\weights\last.pt'
$configDir = Join-Path $repoRoot '.tmp\ultralytics_config_v2'
$runs = @(
    @{ Name = 'center_baseline'; Sequences = 'tracking_scoring_frames_center_v1.csv'; Model = $baseline },
    @{ Name = 'center_oversampled'; Sequences = 'tracking_scoring_frames_center_v1.csv'; Model = $oversampled },
    @{ Name = 'full_baseline'; Sequences = 'tracking_scoring_frames_full_v1.csv'; Model = $baseline },
    @{ Name = 'full_oversampled'; Sequences = 'tracking_scoring_frames_full_v1.csv'; Model = $oversampled }
)

if (-not (Test-Path -LiteralPath $baseline) -or -not (Test-Path -LiteralPath $oversampled)) {
    throw 'Both frozen detector checkpoints are required.'
}
$decision = Get-Content -LiteralPath (Join-Path $repoRoot 'experiments\ball\BALL-HARD-POSITIVE-EXP-0001\holdout_evaluation_v1\tracker_gate_decision_v1.json') -Raw | ConvertFrom-Json
if (-not $decision.gate.passed -or $decision.gate.decision -ne 'REOPEN_TRACKER') {
    throw 'The preregistered detector gate does not authorize tracker retesting.'
}
& $python 'experiments\ball\ORACLE-TRACKER-RETEST-EXP-0001\prepare_tracker_scoring_windows_v1.py'
if ($LASTEXITCODE -ne 0) { throw 'Annotation freeze failed; tracker inference was not started.' }
$env:YOLO_CONFIG_DIR = (Resolve-Path $configDir).Path

foreach ($run in $runs) {
    $output = Join-Path $experiment $run.Name
    if (Test-Path -LiteralPath $output) { throw "Refusing to overwrite prior run output: $output" }
    & $python $runner `
        --sequences (Join-Path $experiment $run.Sequences) `
        --model $run.Model `
        --output $output `
        --confidence 0.01 --max-candidates 10 --imgsz 960 --device cpu
    if ($LASTEXITCODE -ne 0) { throw "Tracker run failed: $($run.Name)" }
}

& $python 'experiments\ball\ORACLE-TRACKER-RETEST-EXP-0001\summarize_tracker_retest_v1.py'
if ($LASTEXITCODE -ne 0) { throw 'Tracker retest summarization failed.' }
