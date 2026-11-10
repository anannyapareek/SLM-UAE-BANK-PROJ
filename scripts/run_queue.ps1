# Overnight grid queue (GPU lab, Windows). Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File D:\slm_project\SLM-UAE-BANK-PROJ\scripts\run_queue.ps1
$ErrorActionPreference = "Continue"
$repo = "D:\slm_project\SLM-UAE-BANK-PROJ"
Set-Location $repo
$py = "$repo\.venv\Scripts\python.exe"
$env:PIP_CACHE_DIR = "D:\slm_cache\pip"; $env:HF_HOME = "D:\slm_cache\hf"
$env:TEMP = "D:\slm_cache\tmp"; $env:TMP = $env:TEMP
$expected = "5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51"
New-Item -ItemType Directory -Force logs | Out-Null
Start-Transcript -Path ("logs\queue_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date)) -Append

function Check-TestSet {
    $h = (Get-FileHash data\processed\test_ids.csv -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $expected) { Write-Host "$(Get-Date -f HH:mm:ss) CHECKSUM FAILED ($h). STOPPING QUEUE."; Stop-Transcript; exit 1 }
    Write-Host "$(Get-Date -f HH:mm:ss) test checksum OK"
}

# priority order: lr 2e-4 r16, 2e-4 r8, 1e-4 r16, 1e-4 r8
$grid = @(@{lr="2e-4"; r=16}, @{lr="2e-4"; r=8}, @{lr="1e-4"; r=16}, @{lr="1e-4"; r=8})

foreach ($g in $grid) {
    $out = "runs\grid_qwen15_r$($g.r)_lr$($g.lr)"
    Write-Host "`n===== $(Get-Date -f HH:mm:ss) $out ====="

    # train (bf16); rerunning resumes from the latest checkpoint
    Check-TestSet
    Remove-Item Env:SLM_SCORE_DTYPE -ErrorAction SilentlyContinue
    if (-not (Test-Path "$out\adapter\adapter_config.json") -or -not (Test-Path "$out\train_done.txt")) {
        & $py -m src.slm.train_lora --device cuda --model primary --models-dir models --data-dir data/slm --out $out --lr $g.lr --rank $g.r --qlora off
        if ($LASTEXITCODE -ne 0) { Write-Host "$(Get-Date -f HH:mm:ss) TRAINING FAILED for $out (exit $LASTEXITCODE); skipping to next config"; continue }
        Set-Content "$out\train_done.txt" (Get-Date -Format s)
    } else { Write-Host "training already done, skipping" }

    # score full validation split, hourly rows, fp16 + merged adapter
    Check-TestSet
    $preds = "$out\preds_valid.parquet"
    if (-not (Test-Path $preds)) {
        $env:SLM_SCORE_DTYPE = "fp16"
        & $py -m src.slm.score --device cuda --model primary --models-dir models --adapter "$out\adapter" --split valid --batch-size 16 --out $preds
        $code = $LASTEXITCODE
        Remove-Item Env:SLM_SCORE_DTYPE -ErrorAction SilentlyContinue
        if ($code -ne 0) { Write-Host "$(Get-Date -f HH:mm:ss) SCORING FAILED for $out (exit $code)"; continue }
    } else { Write-Host "validation already scored, skipping" }

    & $py -c "import pandas as pd; d=pd.read_parquet(r'$preds'); n=int(d.score.isna().sum()); print('rows', len(d), '| NaN', n, '| score min %.4g max %.4g' % (d.score.min(), d.score.max())); print('WARNING: NaN scores, rescore this run in bf16' if n else 'NaN check OK')"
    Write-Host "$(Get-Date -f HH:mm:ss) done $out"
}
Write-Host "`n===== $(Get-Date -f HH:mm:ss) QUEUE FINISHED ====="
Stop-Transcript
