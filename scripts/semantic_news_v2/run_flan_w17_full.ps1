param(
    [ValidateRange(0, 31)]
    [int]$StartShard = 0,
    [ValidateRange(0, 31)]
    [int]$EndShard = 31,
    [switch]$RunPreflight,
    [switch]$Aggregate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$flanPython = Join-Path $repositoryRoot ".venv-flan-t5-xl\Scripts\python.exe"
$trainingPython = Join-Path $repositoryRoot ".venv-training\Scripts\python.exe"
$flanScript = Join-Path $repositoryRoot "scripts\semantic_news_v2\flan_w17.py"
$semanticRoot = Join-Path $repositoryRoot "data\features\news_semantic\massive_v2"
$assignmentPath = Join-Path $semanticRoot "article_target_assignments.jsonl.gz"
$stockDayPath = Join-Path $semanticRoot "stock_day_scope.jsonl.gz"
$corpusManifest = Join-Path $semanticRoot "manifest.json"
$workRoot = Join-Path $semanticRoot "flan_w17"
$preflightPath = Join-Path $workRoot "preflight.json"
$acceptancePath = Join-Path $repositoryRoot "config\flan_w17_acceptance_v1.json"
$dailyOutput = Join-Path $workRoot "daily_wllm17.parquet"

if ($StartShard -gt $EndShard) {
    throw "StartShard must not exceed EndShard."
}
foreach ($requiredPath in @(
    $flanPython,
    $trainingPython,
    $assignmentPath,
    $stockDayPath,
    $corpusManifest,
    $acceptancePath
    $flanScript
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path is missing: $requiredPath"
    }
}
New-Item -ItemType Directory -Path $workRoot -Force | Out-Null

if ($RunPreflight) {
    if (Test-Path -LiteralPath $preflightPath) {
        throw "Preflight already exists; inspect it instead of overwriting: $preflightPath"
    }
    & $flanPython $flanScript preflight `
        --input $assignmentPath `
        --output $preflightPath
    if ($LASTEXITCODE -ne 0) {
        throw "FLAN W17 tokenizer preflight failed."
    }
}
if (-not (Test-Path -LiteralPath $preflightPath)) {
    throw "A completed preflight is required: $preflightPath"
}

for ($shard = $StartShard; $shard -le $EndShard; $shard++) {
    $predictionPath = Join-Path $workRoot ("predictions-shard-{0:D2}.jsonl" -f $shard)
    & $flanPython $flanScript infer `
        --input $assignmentPath `
        --output $predictionPath `
        --acceptance-config $acceptancePath `
        --preflight-manifest $preflightPath `
        --shard-index $shard `
        --shard-count 32
    if ($LASTEXITCODE -ne 0) {
        throw "FLAN W17 inference failed on shard $shard."
    }
}

if ($Aggregate) {
    if ($StartShard -ne 0 -or $EndShard -ne 31) {
        throw "Aggregation is permitted only after requesting all 32 shards."
    }
    $predictionPaths = 0..31 | ForEach-Object {
        Join-Path $workRoot ("predictions-shard-{0:D2}.jsonl" -f $_)
    }
    $aggregateArguments = @(
        $flanScript,
        "aggregate",
        "--assignments",
        $assignmentPath,
        "--predictions"
    ) + $predictionPaths + @(
        "--stock-days",
        $stockDayPath,
        "--corpus-manifest",
        $corpusManifest,
        "--preflight-manifest",
        $preflightPath,
        "--acceptance-config",
        $acceptancePath,
        "--output",
        $dailyOutput
    )
    & $trainingPython @aggregateArguments
    if ($LASTEXITCODE -ne 0) {
        throw "FLAN W17 daily aggregation failed."
    }
}
