param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('flan-t5-xl', 'llama-3.1')]
    [string]$Runner,

    [Parameter(Mandatory = $true)]
    [string]$LogPath
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repositoryRoot 'data\runtime\python311-embed\base\python.exe'
$launcher = Join-Path $repositoryRoot 'scripts\run_with_isolated_python311.py'
$preflight = Join-Path $repositoryRoot 'scripts\preflight_news_text_ablation_inputs.py'
$benchmarkRoot = Join-Path $repositoryRoot 'data\benchmarks\news_text_ablation_300\v1'
$outputRoot = Join-Path $repositoryRoot 'outputs\news_provider_fulltext\v1_0'
$resolvedLogPath = [System.IO.Path]::GetFullPath($LogPath)
New-Item -ItemType Directory -Force -Path (
    Split-Path -Parent $resolvedLogPath
) | Out-Null

if ($Runner -eq 'flan-t5-xl') {
    $sitePackages = Join-Path $repositoryRoot '.venv-flan-t5-xl\Lib\site-packages'
    $output = Join-Path $outputRoot 'flan_context_preflight.json'
}
else {
    $sitePackages = Join-Path $repositoryRoot '.venv-llama31\Lib\site-packages'
    $output = Join-Path $outputRoot 'llama_context_preflight.json'
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python $launcher `
    --site-packages $sitePackages `
    --script $preflight -- `
    --runner $Runner `
    --input (Join-Path $benchmarkRoot 'massive_description.jsonl') `
    --input (Join-Path $benchmarkRoot 'alpha_summary.jsonl') `
    --input (Join-Path $benchmarkRoot 'fulltext_evidence_chunks.jsonl') `
    --output $output *> $resolvedLogPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference

if ($exitCode -ne 0) {
    throw "$Runner tokenizer preflight failed with exit code $exitCode"
}
Write-Output "Completed $Runner tokenizer preflight -> $output"
