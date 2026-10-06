$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repoRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$experiment = Join-Path $repoRoot 'experiments\ball\BALL-HARD-POSITIVE-EXP-0001'
$baselineMeta = Join-Path $experiment 'train\baseline\metrics.json'
$baselineWeights = Join-Path $repoRoot 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\baseline\weights\last.pt'
$oversampledMeta = Join-Path $experiment 'train\oversampled\metrics.json'
$oversampledWeights = Join-Path $repoRoot 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\oversampled\weights\last.pt'
$log = Join-Path $experiment 'sequential_followup.log'
$configDir = Join-Path $repoRoot '.tmp\ultralytics_config_v2'
$deadline = [DateTime]::UtcNow.AddHours(8)

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

try {
    $env:YOLO_CONFIG_DIR = (Resolve-Path $configDir).Path
    Write-Log 'Queue active; waiting for baseline metrics.json with status=completed.'
    while (-not (Test-Path -LiteralPath $baselineMeta)) {
        if ([DateTime]::UtcNow -ge $deadline) { throw 'Timed out after 8 hours waiting for baseline completion.' }
        Start-Sleep -Seconds 30
    }
    $baseline = Get-Content -LiteralPath $baselineMeta -Raw | ConvertFrom-Json
    if ($baseline.status -ne 'completed' -or -not (Test-Path -LiteralPath $baselineWeights)) {
        throw 'Baseline metadata or last.pt does not confirm a completed run.'
    }
    Write-Log 'Baseline completion confirmed; starting fixed oversampled training.'

    & $python 'experiments\ball\run_ball_exp0001.py' `
        --data 'experiments\ball\BALL-HARD-POSITIVE-EXP-0001\dataset_small_positive_oversampled.yaml' `
        --model 'yolo11n.pt' `
        --output 'experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\oversampled' `
        --epochs 30 --imgsz 960 --batch 8 --device cpu --no-val --no-amp *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Oversampled training exited with code $LASTEXITCODE." }

    if (-not (Test-Path -LiteralPath $oversampledMeta) -or -not (Test-Path -LiteralPath $oversampledWeights)) {
        throw 'Oversampled run ended without completion metadata or last.pt.'
    }
    $oversampled = Get-Content -LiteralPath $oversampledMeta -Raw | ConvertFrom-Json
    if ($oversampled.status -ne 'completed') { throw 'Oversampled metadata status is not completed.' }
    Write-Log 'Oversampled completion confirmed; running the frozen holdout comparison once.'

    & $python 'experiments\ball\BALL-HARD-POSITIVE-EXP-0001\evaluate_holdout_v1.py' `
        --baseline 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\baseline\weights\last.pt' `
        --oversampled 'runs\detect\experiments\ball\BALL-HARD-POSITIVE-EXP-0001\train\oversampled\weights\last.pt' *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Frozen holdout evaluation exited with code $LASTEXITCODE." }
    Write-Log 'Frozen holdout evaluation completed; applying the preregistered tracker gate.'

    & $python 'experiments\ball\BALL-HARD-POSITIVE-EXP-0001\decide_tracker_gate_v1.py' *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Frozen tracker-gate decision exited with code $LASTEXITCODE." }
    Write-Log 'Sequential detector comparison and gate decision finished successfully.'
    exit 0
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    exit 1
}
