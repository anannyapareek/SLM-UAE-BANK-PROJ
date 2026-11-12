# Night-2 queue (GPU lab, Windows PowerShell 5.1): Comparison 1 seeds, optional Comparison 2.
#
# Phase A  train seeds 2 and 3 of the Gate 2 config (lr 1e-4, rank 8) on data/slm, one at a time.
# Phase B  Comparison 1 scoring: seed 1 (grid adapter) on calib and test; seeds 2 and 3 on
#          valid, calib and test. Up to -Parallel score.py processes at once, largest first.
# Phase C  (only with -IncludeText) Comparison 2: train seeds 1-3 on data/slm_text, then score
#          valid, calib and test with --text --only-final --batch-size 8.
# The script writes predictions only: it never computes or prints metrics. Training never
# overlaps scoring. The frozen test SHA-256 is checked before every job; a mismatch stops
# everything. Any other failed job is recorded and the queue carries on.
#
# Run:
#   powershell -ExecutionPolicy Bypass -File D:\slm_project\SLM-UAE-BANK-PROJ\scripts\run_night2.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File D:\slm_project\SLM-UAE-BANK-PROJ\scripts\run_night2.ps1 -Parallel 2
#   options: -Parallel 1|2|3 (default 1)  -Comp1Granularity hourly|final (default hourly)
#            -IncludeText (add Comparison 2)  -DryRun (list the jobs and exit)
# Rerun after a failure: run the same command again. Finished training (adapter +
# train_done.txt) and existing prediction files are skipped; an interrupted training run
# resumes from its last checkpoint. Delete a bad predictions file to have it rescored.
# Logs: logs\night2\summary.txt (every job: start, end, exit status, checks) and one
# .out.log / .err.log per job (python output; Start-Transcript does not capture it).
param(
    [ValidateSet(1, 2, 3)][int]$Parallel = 1,
    [ValidateSet("hourly", "final")][string]$Comp1Granularity = "hourly",
    [switch]$IncludeText,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$repo = "D:\slm_project\SLM-UAE-BANK-PROJ"
Set-Location $repo
$py = "$repo\.venv\Scripts\python.exe"
$env:PIP_CACHE_DIR = "D:\slm_cache\pip"; $env:HF_HOME = "D:\slm_cache\hf"
$env:TEMP = "D:\slm_cache\tmp"; $env:TMP = $env:TEMP
$env:PYTHONUNBUFFERED = "1"
$expected = "5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51"
$logDir = "logs\night2"
$summary = "$logDir\summary.txt"
New-Item -ItemType Directory -Force $logDir, runs | Out-Null
Start-Transcript -Path ("logs\night2_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date)) -Append

$lr = "1e-4"; $rank = 8
$seed1Adapter = "runs\grid_qwen15_r8_lr1e-4\adapter"
$common = @("--device", "cuda", "--model", "primary", "--models-dir", "models")
$expectedRows = @{ valid = 336737; calib = 309413; test = 848137 }
$splitOrder = @("test", "valid", "calib")      # largest first
$script:failed = @()
$script:running = @()

function Write-Summary([string]$msg) {
    $line = "$(Get-Date -Format s) | $msg"
    Write-Host $line
    if (-not $DryRun) { Add-Content -Path $summary -Value $line }
}

function Stop-Queue([string]$reason) {
    Write-Summary "STOPPING QUEUE: $reason"
    foreach ($r in $script:running) {
        if (-not $r.Proc.HasExited) {
            Stop-Process -Id $r.Proc.Id -Force -ErrorAction SilentlyContinue
            Write-Summary "killed $($r.Job.Name)"
        }
    }
    Write-Summary "NIGHT2 STOPPED"
    Stop-Transcript
    exit 1
}

function Assert-TestSet([string]$before) {
    $h = (Get-FileHash data\processed\test_ids.csv -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $expected) { Stop-Queue "test_ids.csv SHA-256 $h does not match the frozen hash (before $before)" }
    Write-Summary "test checksum OK (before $before)"
}

function Start-Python([string]$name, [string[]]$argList) {
    $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
        -RedirectStandardOutput "$logDir\$name.out.log" -RedirectStandardError "$logDir\$name.err.log"
    $null = $p.Handle    # keeps ExitCode available after the process ends (PS 5.1)
    return $p
}

function Get-Duration($start) {
    return "{0:N1} min" -f ((Get-Date) - $start).TotalMinutes
}

# ---------- job lists ----------

function New-TrainJob([string]$name, [string]$runDir, [int]$seed, [string]$dataDir) {
    $argList = @("-m", "src.slm.train_lora") + $common + @("--data-dir", $dataDir, "--out", $runDir,
        "--lr", $lr, "--rank", "$rank", "--seed", "$seed", "--qlora", "off", "--log", "$runDir\train.log")
    return [pscustomobject]@{ Name = $name; RunDir = $runDir; Args = $argList }
}

function New-ScoreJob([string]$name, [string]$adapter, [string]$split, [string]$out,
                      [bool]$final, [string[]]$extra, [string]$batch) {
    $argList = @("-m", "src.slm.score") + $common + @("--adapter", $adapter, "--split", $split,
        "--batch-size", $batch, "--out", $out, "--log", "$logDir\$name.score.log")
    if ($final) { $argList += "--only-final" }
    if ($split -eq "test") { $argList += "--allow-test" }
    $argList += $extra
    return [pscustomobject]@{ Name = $name; Adapter = $adapter; Split = $split; Out = $out;
                              Final = $final; Args = $argList }
}

function Get-ScoreJobs([string]$prefix, [hashtable]$seedPlan, [bool]$final, [string[]]$extra,
                       [string]$batch) {
    # seedPlan: seed -> @{ Adapter = ...; RunDir = ...; Splits = @(...) }
    $jobs = @()
    $suffix = ""
    if ($final) { $suffix = "_final" }
    foreach ($split in $splitOrder) {
        foreach ($seed in ($seedPlan.Keys | Sort-Object)) {
            $plan = $seedPlan[$seed]
            if ($plan.Splits -contains $split) {
                $out = "$($plan.RunDir)\preds_$split$suffix.parquet"
                $jobs += New-ScoreJob "$($prefix)_s$($seed)_$split$suffix" $plan.Adapter $split $out $final $extra $batch
            }
        }
    }
    return $jobs
}

# ---------- runners ----------

function Invoke-Train($job) {
    if ((Test-Path "$($job.RunDir)\adapter\adapter_config.json") -and (Test-Path "$($job.RunDir)\train_done.txt")) {
        Write-Summary "SKIP $($job.Name): adapter and train_done.txt exist"
        return $true
    }
    Assert-TestSet $job.Name
    Remove-Item Env:SLM_SCORE_DTYPE -ErrorAction SilentlyContinue    # training stays bf16
    New-Item -ItemType Directory -Force $job.RunDir | Out-Null
    $start = Get-Date
    Write-Summary "START $($job.Name) | python $($job.Args -join ' ')"
    $p = Start-Python $job.Name $job.Args
    $p.WaitForExit()
    $code = $p.ExitCode
    $ok = ($code -eq 0) -and (Test-Path "$($job.RunDir)\adapter\adapter_config.json")
    if ($ok) {
        Set-Content "$($job.RunDir)\train_done.txt" (Get-Date -Format s)
        Write-Summary "END $($job.Name) | exit $code | OK | $(Get-Duration $start)"
    } else {
        Write-Summary "END $($job.Name) | exit $code | FAILED | $(Get-Duration $start) | see $logDir\$($job.Name).err.log"
        $script:failed += $job.Name
    }
    return $ok
}

function Complete-ScoreJob($r) {
    $code = $r.Proc.ExitCode
    if ($code -ne 0) {
        Write-Summary "END $($r.Job.Name) | exit $code | FAILED | $(Get-Duration $r.Start) | see $logDir\$($r.Job.Name).err.log"
        $script:failed += $r.Job.Name
        return
    }
    Write-Summary "END $($r.Job.Name) | exit 0 | $(Get-Duration $r.Start)"
    $checkArgs = @("scripts\check_preds.py", "--preds", $r.Job.Out, "--split", $r.Job.Split)
    if ($r.Job.Final) { $checkArgs += "--final" }
    $check = (& $py @checkArgs 2>&1 | Out-String).Trim()
    $checkCode = $LASTEXITCODE
    $note = ""
    if (-not $r.Job.Final) { $note = " | expected hourly rows $($expectedRows[$r.Job.Split])" }
    Write-Summary "$check$note"
    if ($checkCode -ne 0) { $script:failed += "$($r.Job.Name) (check)" }
}

function Invoke-ScoreQueue($jobs) {
    $pending = New-Object System.Collections.ArrayList
    foreach ($j in $jobs) { [void]$pending.Add($j) }
    $script:running = @()
    while (($pending.Count -gt 0) -or ($script:running.Count -gt 0)) {
        # collect finished jobs
        $still = @()
        foreach ($r in $script:running) {
            if ($r.Proc.HasExited) { Complete-ScoreJob $r } else { $still += $r }
        }
        $script:running = $still
        # start new jobs up to -Parallel
        while (($pending.Count -gt 0) -and ($script:running.Count -lt $Parallel)) {
            $job = $pending[0]
            $pending.RemoveAt(0)
            if (Test-Path $job.Out) {
                Write-Summary "SKIP $($job.Name): $($job.Out) exists"
                continue
            }
            if (-not (Test-Path "$($job.Adapter)\adapter_config.json")) {
                Write-Summary "SKIP $($job.Name): no adapter at $($job.Adapter) (training failed or missing)"
                $script:failed += "$($job.Name) (no adapter)"
                continue
            }
            Assert-TestSet $job.Name
            $env:SLM_SCORE_DTYPE = "fp16"    # scoring only; score.py merges the adapter
            $outDir = Split-Path $job.Out -Parent
            New-Item -ItemType Directory -Force $outDir | Out-Null
            Write-Summary "START $($job.Name) | python $($job.Args -join ' ')"
            $p = Start-Python $job.Name $job.Args
            $script:running += [pscustomobject]@{ Job = $job; Proc = $p; Start = (Get-Date) }
        }
        if (($pending.Count -gt 0) -or ($script:running.Count -gt 0)) { Start-Sleep -Seconds 20 }
    }
    Remove-Item Env:SLM_SCORE_DTYPE -ErrorAction SilentlyContinue
}

# ---------- plan ----------

$c1Final = ($Comp1Granularity -eq "final")
$trainA = @(
    (New-TrainJob "train_c1_s2" "runs\c1_qwen15_r8_lr1e-4_s2" 2 "data/slm"),
    (New-TrainJob "train_c1_s3" "runs\c1_qwen15_r8_lr1e-4_s3" 3 "data/slm")
)
$c1Plan = @{
    1 = @{ Adapter = $seed1Adapter; RunDir = "runs\c1_qwen15_r8_lr1e-4_s1"; Splits = @("calib", "test") }
    2 = @{ Adapter = "runs\c1_qwen15_r8_lr1e-4_s2\adapter"; RunDir = "runs\c1_qwen15_r8_lr1e-4_s2"; Splits = @("valid", "calib", "test") }
    3 = @{ Adapter = "runs\c1_qwen15_r8_lr1e-4_s3\adapter"; RunDir = "runs\c1_qwen15_r8_lr1e-4_s3"; Splits = @("valid", "calib", "test") }
}
$scoreB = @(Get-ScoreJobs "score_c1" $c1Plan $c1Final @() "16")

$trainC = @()
$scoreC = @()
if ($IncludeText) {
    $c2Plan = @{}
    foreach ($s in 1, 2, 3) {
        $dir = "runs\c2_qwen15_r8_lr1e-4_s$s"
        $trainC += New-TrainJob "train_c2_s$s" $dir $s "data/slm_text"
        $c2Plan[$s] = @{ Adapter = "$dir\adapter"; RunDir = $dir; Splits = @("valid", "calib", "test") }
    }
    $scoreC = @(Get-ScoreJobs "score_c2" $c2Plan $true @("--text") "8")
}

Write-Summary "===== NIGHT2 START | Parallel $Parallel | Comparison 1 $Comp1Granularity | IncludeText $IncludeText | DryRun $DryRun ====="

if ($DryRun) {
    $groups = @(
        @{ Title = "Phase A: training (one at a time)"; Jobs = $trainA; Train = $true },
        @{ Title = "Phase B: Comparison 1 scoring (up to $Parallel at once)"; Jobs = $scoreB; Train = $false },
        @{ Title = "Phase C: Comparison 2 training"; Jobs = $trainC; Train = $true },
        @{ Title = "Phase C: Comparison 2 scoring (up to $Parallel at once)"; Jobs = $scoreC; Train = $false }
    )
    foreach ($g in $groups) {
        if ($g.Jobs.Count -eq 0) { continue }
        Write-Host "`n$($g.Title)"
        foreach ($j in $g.Jobs) {
            $state = "would run"
            if ($g.Train) {
                if ((Test-Path "$($j.RunDir)\adapter\adapter_config.json") -and (Test-Path "$($j.RunDir)\train_done.txt")) { $state = "would skip (done)" }
                Write-Host "  [$state] $($j.Name): python $($j.Args -join ' ')"
            } else {
                if (Test-Path $j.Out) { $state = "would skip (exists)" }
                $envNote = "SLM_SCORE_DTYPE=fp16"
                Write-Host "  [$state] $($j.Name): $envNote python $($j.Args -join ' ')"
            }
        }
    }
    if (-not $IncludeText) { Write-Host "`nPhase C: not requested (-IncludeText)" }
    Write-Host "`nDry run only: nothing was started."
    Stop-Transcript
    exit 0
}

# ---------- Phase A ----------
Write-Summary "===== PHASE A: training Comparison 1 seeds 2 and 3 ====="
foreach ($job in $trainA) { [void](Invoke-Train $job) }

# ---------- Phase B ----------
Write-Summary "===== PHASE B: Comparison 1 scoring ($($scoreB.Count) jobs, $Comp1Granularity) ====="
Invoke-ScoreQueue $scoreB

# ---------- Phase C ----------
if ($IncludeText) {
    Write-Summary "===== PHASE C: Comparison 2 (text, logoff-only) ====="
    $missing = @()
    if (-not (Test-Path "data\slm_text\train.jsonl")) { $missing += "data\slm_text\train.jsonl (build it with make_train --text)" }
    if (-not (Test-Path "data\slm_text\valid.jsonl")) { $missing += "data\slm_text\valid.jsonl (build it with make_train --text)" }
    $scoreHelp = (& $py -m src.slm.score --help 2>&1 | Out-String)
    if ($scoreHelp -notmatch "--text") { $missing += "the --text option in src/slm/score.py" }
    $makeHelp = (& $py -m src.slm.make_train --help 2>&1 | Out-String)
    if ($makeHelp -notmatch "--text") { $missing += "the --text option in src/slm/make_train.py" }
    if ($missing.Count -gt 0) {
        Write-Summary "PHASE C NOT RUN: missing $($missing -join '; ')"
        $script:failed += "Phase C (prerequisites missing)"
    } else {
        foreach ($job in $trainC) { [void](Invoke-Train $job) }
        Invoke-ScoreQueue $scoreC
    }
}

# ---------- end ----------
Write-Summary "===== NIGHT2 FINISHED ====="
if ($script:failed.Count -eq 0) {
    Write-Summary "failed jobs: none"
} else {
    Write-Summary "failed jobs ($($script:failed.Count)): $($script:failed -join ', ')"
}
Write-Host "NIGHT2 FINISHED"
Stop-Transcript
