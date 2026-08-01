$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = (
    Resolve-Path (Join-Path $PSScriptRoot "..\..")
).Path
$python = Join-Path $repositoryRoot ".venv-training\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Training Python is missing: $python"
}

Push-Location $repositoryRoot
try {
    & $python -m scripts.quant_deterministic_news_training_v3.lock_protocol
    if ($LASTEXITCODE -ne 0) {
        throw "V3 protocol lock failed."
    }

    foreach ($bundle in @("S0", "S1", "S2", "S3")) {
        & $python -m scripts.quant_deterministic_news_training_v3.run_linear `
            --bundle $bundle `
            --allow-failed-gate-exploratory
        if ($LASTEXITCODE -ne 0) {
            throw "V3 linear bundle $bundle failed."
        }
    }

    foreach ($bundle in @(
        "C-S2-COV",
        "C-S3-COV",
        "C-S2-L20",
        "C-S3-L20",
        "C-S2-WS",
        "C-S3-WS",
        "C-S2-PERM",
        "C-S3-PERM",
        "C-S2-LONG",
        "C-S3-LONG"
    )) {
        & $python -m scripts.quant_deterministic_news_training_v3.run_linear `
            --bundle $bundle `
            --allow-failed-gate-exploratory
        if ($LASTEXITCODE -ne 0) {
            throw "V3 control bundle $bundle failed."
        }
    }

    & $python -m scripts.quant_deterministic_news_training_v3.run_xgboost `
        --allow-failed-gate-exploratory `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "V3 XGBoost bundle failed."
    }

    & $python `
        -m scripts.quant_deterministic_news_training_v3.run_final_comparison `
        --allow-failed-gate-exploratory
    if ($LASTEXITCODE -ne 0) {
        throw "V3 final comparison failed."
    }
}
finally {
    Pop-Location
}
