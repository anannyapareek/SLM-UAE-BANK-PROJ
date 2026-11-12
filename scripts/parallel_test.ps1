# Parallel scoring test (GPU lab, Windows PowerShell 5.1), about 10 minutes.
# Scores the same 10,000 validation rows (score.py --rows 10000, a fixed seed-1 sample) with the
# seed-1 adapter in fp16, running 1, then 2, then 3 score.py processes at once, and reports
# throughput. Writes results\parallel_test.json; temporary predictions are deleted afterwards.
#
# Run (nothing else may be using the GPU, i.e. no training running):
#   powershell -ExecutionPolicy Bypass -File D:\slm_project\SLM-UAE-BANK-PROJ\scripts\parallel_test.ps1
# After a failure: read logs\night2\parallel_*.out.log / .err.log, fix, and simply rerun;
# the script always starts from scratch.
$ErrorActionPreference = "Continue"
$repo = "D:\slm_project\SLM-UAE-BANK-PROJ"
Set-Location $repo
$py = "$repo\.venv\Scripts\python.exe"
$env:PIP_CACHE_DIR = "D:\slm_cache\pip"; $env:HF_HOME = "D:\slm_cache\hf"
$env:TEMP = "D:\slm_cache\tmp"; $env:TMP = $env:TEMP
$env:PYTHONUNBUFFERED = "1"
$expected = "5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51"
$adapter = "runs\grid_qwen15_r8_lr1e-4\adapter"
$rows = 10000
$logDir = "logs\night2"
$tmpDir = "runs\parallel_test_tmp"
New-Item -ItemType Directory -Force $logDir, $tmpDir, results | Out-Null
$logFile = "$logDir\parallel_test.txt"
Start-Transcript -Path ("logs\parallel_test_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date)) -Append

function Write-Log([string]$msg) {
    $line = "$(Get-Date -Format s) $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

function Assert-TestSet {
    $h = (Get-FileHash data\processed\test_ids.csv -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $expected) {
        Write-Log "CHECKSUM FAILED for data\processed\test_ids.csv ($h). STOPPING."
        Stop-Transcript
        exit 1
    }
    Write-Log "test checksum OK"
}

if (-not (Test-Path "$adapter\adapter_config.json")) {
    Write-Log "Seed-1 adapter not found at $adapter. STOPPING."
    Stop-Transcript
    exit 1
}

$results = @()
foreach ($n in 1, 2, 3) {
    Assert-TestSet
    Write-Log "== $n process(es) at once, $rows rows each"
    $env:SLM_SCORE_DTYPE = "fp16"
    $procs = @()
    $start = Get-Date
    for ($i = 1; $i -le $n; $i++) {
        $tag = "n$($n)_p$i"
        $out = "$tmpDir\preds_$tag.parquet"
        $pyLog = "$logDir\parallel_$tag.score.log"
        $argList = @("-m", "src.slm.score", "--device", "cuda", "--model", "primary",
                     "--models-dir", "models", "--adapter", $adapter, "--split", "valid",
                     "--rows", "$rows", "--batch-size", "16", "--out", $out, "--log", $pyLog)
        $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
            -RedirectStandardOutput "$logDir\parallel_$tag.out.log" `
            -RedirectStandardError "$logDir\parallel_$tag.err.log"
        $null = $p.Handle    # keeps ExitCode available after the process ends (PS 5.1)
        $procs += [pscustomobject]@{ Tag = $tag; Proc = $p; Log = $pyLog }
    }
    foreach ($item in $procs) { $item.Proc.WaitForExit() }
    $wall = ((Get-Date) - $start).TotalSeconds
    Remove-Item Env:SLM_SCORE_DTYPE -ErrorAction SilentlyContinue

    $perProcess = @()
    $allOk = $true
    foreach ($item in $procs) {
        $code = $item.Proc.ExitCode
        $seconds = $null
        if (Test-Path $item.Log) {
            $m = Select-String -Path $item.Log -Pattern "scored (\d+) rows in ([0-9.]+) s" | Select-Object -Last 1
            if ($m) { $seconds = [double]$m.Matches[0].Groups[2].Value }
        }
        if ($code -ne 0 -or $seconds -eq $null) { $allOk = $false }
        $rate = $null
        if ($seconds) { $rate = [math]::Round($rows / $seconds, 1) }
        $perProcess += [pscustomobject]@{ process = $item.Tag; exit_code = $code;
                                          scoring_seconds = $seconds; rows_per_second = $rate }
        Write-Log ("  {0}: exit {1}, scored {2} rows in {3} s ({4} rows/s, excl. model loading)" -f `
                   $item.Tag, $code, $rows, $seconds, $rate)
    }
    $total = [math]::Round($n * $rows / $wall, 1)
    Write-Log ("  N={0}: wall-clock {1:N1} s (incl. model loading), total {2} rows/s; all ok: {3}" -f `
               $n, $wall, $total, $allOk)
    $results += [pscustomobject]@{ processes = $n; rows_per_process = $rows;
                                   wall_clock_seconds = [math]::Round($wall, 1);
                                   total_rows_per_second = $total; all_ok = $allOk;
                                   per_process = $perProcess }
}

Write-Log "== Summary (total rows/s = N x $rows / wall-clock, which includes model loading)"
$table = $results | Select-Object processes, wall_clock_seconds, total_rows_per_second, all_ok,
    @{ Name = "per_process_rows_per_second"; Expression = { ($_.per_process | ForEach-Object { $_.rows_per_second }) -join ", " } } |
    Format-Table -AutoSize | Out-String
Write-Host $table
Add-Content -Path $logFile -Value $table

$report = [pscustomobject]@{ created = (Get-Date -Format s); adapter = $adapter; split = "valid";
                             rows = $rows; batch_size = 16; dtype = "fp16"; runs = $results }
$report | ConvertTo-Json -Depth 5 | Set-Content -Path results\parallel_test.json -Encoding ASCII
Write-Log "wrote results\parallel_test.json"

Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Log "deleted temporary predictions in $tmpDir"
Write-Log "PARALLEL TEST FINISHED"
Stop-Transcript
