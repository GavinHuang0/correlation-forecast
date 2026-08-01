param(
    [ValidateRange(0, 50488)]
    [int]$MaxNewArticles = 0,
    [switch]$PrepareOnly,
    [switch]$PreflightOnly,
    [switch]$StatusOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainingPython = Join-Path $repositoryRoot ".venv-training\Scripts\python.exe"
$flanPython = Join-Path $repositoryRoot ".venv-flan-t5-xl\Scripts\python.exe"
$runner = Join-Path $repositoryRoot "scripts\semantic_news_v3\flan_w17_lite.py"
$outputRoot = Join-Path $repositoryRoot "data\features\news_semantic\massive_v3\flan_w17_lite_k16"

foreach ($requiredPath in @($trainingPython, $flanPython, $runner)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path is missing: $requiredPath"
    }
}

Push-Location $repositoryRoot
try {
    if ($StatusOnly) {
        & $trainingPython $runner status --output-root $outputRoot
        if ($LASTEXITCODE -ne 0) {
            throw "W17-Lite status validation failed."
        }
        return
    }

    & $trainingPython $runner prepare --output-root $outputRoot | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "W17-Lite K-16 selection preparation failed."
    }
    Write-Host "K-16 selection contract validated (50,488 unique articles)."
    if ($PrepareOnly) {
        return
    }

    $queue = Join-Path $outputRoot "selected_articles.jsonl.gz"
    $selectionManifest = Join-Path $outputRoot "selection_manifest.json"
    $preflightManifest = Join-Path $outputRoot "preflight.json"
    & $flanPython $runner preflight `
        --queue $queue `
        --selection-manifest $selectionManifest `
        --output $preflightManifest | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "W17-Lite tokenizer preflight failed."
    }
    Write-Host "Tokenizer preflight validated (100,976 prompts; no truncation)."
    if ($PreflightOnly) {
        return
    }

    $predictionPath = Join-Path $outputRoot "predictions.jsonl"
    $inferenceArguments = @(
        $runner,
        "infer",
        "--queue",
        $queue,
        "--selection-manifest",
        $selectionManifest,
        "--preflight-manifest",
        $preflightManifest,
        "--output",
        $predictionPath,
        "--recover-stale-lock"
    )
    if ($MaxNewArticles -gt 0) {
        $inferenceArguments += @("--max-new-articles", "$MaxNewArticles")
    }
    & $flanPython @inferenceArguments
    $inferenceExitCode = $LASTEXITCODE
    if ($inferenceExitCode -eq 130 -or $inferenceExitCode -eq 143) {
        Write-Host "W17-Lite stopped safely. Run this same command again to continue."
        exit $inferenceExitCode
    }
    if ($inferenceExitCode -ne 0) {
        throw "W17-Lite inference failed with exit code $inferenceExitCode."
    }
}
finally {
    Pop-Location
}
