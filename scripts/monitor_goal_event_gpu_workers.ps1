param(
    [string]$WorkerPidsCsv,
    [int]$RunnerPid,
    [string]$Output,
    [int]$IntervalSeconds = 5
)

$ErrorActionPreference = 'Stop'
$WorkerPid = @($WorkerPidsCsv -split ',' | ForEach-Object { [int]$_ })
$samplesPath = Join-Path $Output 'usage_samples_actual_workers.csv'
$rows = @()
while (@($WorkerPid | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
    $now = Get-Date
    $gpuByPid = @{}
    try {
        $counters = (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples
        foreach ($workerProcessId in $WorkerPid) {
            $pattern = "pid_${workerProcessId}_"
            $gpuByPid[$workerProcessId] = [math]::Round((@($counters | Where-Object { $_.InstanceName -like "*$pattern*" } | Measure-Object -Property CookedValue -Sum).Sum), 3)
        }
    } catch {
        # O CSV continua registrando CPU e memória se o contador de GPU não estiver disponível.
    }
    foreach ($workerProcessId in $WorkerPid) {
        $process = Get-Process -Id $workerProcessId -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        $row = [PSCustomObject]@{
            timestamp = $now.ToString('o')
            pid = $workerProcessId
            cpu_seconds = [math]::Round($process.CPU, 3)
            working_set_mb = [math]::Round($process.WorkingSet64 / 1MB, 2)
            gpu_engine_utilization_sum_percent = if ($gpuByPid.ContainsKey($workerProcessId)) { $gpuByPid[$workerProcessId] } else { $null }
        }
        $rows += $row
        $row | Export-Csv -NoTypeInformation -Append -Encoding utf8 $samplesPath
    }
    Start-Sleep -Seconds $IntervalSeconds
}
if ($rows.Count) {
    $gpu = @($rows | Where-Object { $null -ne $_.gpu_engine_utilization_sum_percent } | Select-Object -ExpandProperty gpu_engine_utilization_sum_percent)
    [ordered]@{
        worker_pids = $WorkerPid
        samples = $rows.Count
        gpu_engine_utilization_sum_mean_percent = if ($gpu.Count) { [math]::Round((($gpu | Measure-Object -Average).Average), 3) } else { $null }
        gpu_engine_utilization_sum_max_percent = if ($gpu.Count) { [math]::Round((($gpu | Measure-Object -Maximum).Maximum), 3) } else { $null }
    } | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Output 'usage_actual_workers_summary.json')
}
