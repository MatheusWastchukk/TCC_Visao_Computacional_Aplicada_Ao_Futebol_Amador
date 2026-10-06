$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repoRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$experiment = Join-Path $repoRoot 'experiments\ball\ORACLE-TRACKER-RETEST-EXP-0001'
$configDir = Join-Path $repoRoot '.tmp\ultralytics_config_v2'
$runs = @(
    @{ Name = 'center_baseline'; Input = 'blind_tracker_inputs_center_v1.csv'; Model = 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\baseline\weights\last.pt' },
    @{ Name = 'center_oversampled'; Input = 'blind_tracker_inputs_center_v1.csv'; Model = 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\oversampled\weights\last.pt' },
    @{ Name = 'full_baseline'; Input = 'blind_tracker_inputs_full_v1.csv'; Model = 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\baseline\weights\last.pt' },
    @{ Name = 'full_oversampled'; Input = 'blind_tracker_inputs_full_v1.csv'; Model = 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\oversampled\weights\last.pt' }
)
$freezePath = Join-Path $experiment 'blind_tracker_input_freeze_v1.json'
if (-not (Test-Path -LiteralPath $freezePath)) { throw 'Run prepare_blind_tracker_inputs_v1.py first.' }
$freeze = Get-Content -LiteralPath $freezePath -Raw | ConvertFrom-Json
if (-not $freeze.prediction_only -or $freeze.labels_included) { throw 'Blind input freeze is invalid.' }
$gate = Get-Content -LiteralPath (Join-Path $repoRoot 'experiments\ball\BALL-HARD-POSITIVE-EXP-0001\holdout_evaluation_v1\tracker_gate_decision_v1.json') -Raw | ConvertFrom-Json
if (-not $gate.gate.passed -or $gate.gate.decision -ne 'REOPEN_TRACKER') { throw 'Detector gate does not authorize this retest.' }
$env:YOLO_CONFIG_DIR = (Resolve-Path $configDir).Path
foreach ($run in $runs) {
    $output = Join-Path $experiment $run.Name
    if (Test-Path -LiteralPath $output) { throw "Refusing to overwrite existing run output: $output" }
    $modelPath = Join-Path $repoRoot $run.Model
    if (-not (Test-Path -LiteralPath $modelPath)) { throw "Missing detector checkpoint: $modelPath" }
    & $python 'experiments\ball\run_ball_multitrack_pilot.py' `
        --sequences (Join-Path $experiment $run.Input) `
        --model $modelPath --output $output `
        --confidence 0.01 --max-candidates 10 --imgsz 960 --device cpu --prediction-only
    if ($LASTEXITCODE -ne 0) { throw "Blind tracker inference failed: $($run.Name)" }
}
& $python 'experiments\ball\ORACLE-TRACKER-RETEST-EXP-0001\freeze_blind_tracker_runs_v1.py'
if ($LASTEXITCODE -ne 0) { throw 'Blind inference manifest validation failed.' }
Write-Output 'Blind inference complete. Do not inspect against annotations until manual annotation freeze.'
