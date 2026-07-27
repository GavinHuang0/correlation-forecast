param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('flan-t5-xl', 'llama-3.1')]
    [string]$Runner,

    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'massive_description',
        'alpha_summary',
        'fulltext_evidence_chunks'
    )]
    [string]$Variant,

    [int]$FlanBatchSize = 32,

    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repositoryRoot 'data\runtime\python311-embed\base\python.exe'
$launcher = Join-Path $repositoryRoot 'scripts\run_with_isolated_python311.py'
$input = Join-Path $repositoryRoot (
    "data\benchmarks\news_text_ablation_300\v1\$Variant.jsonl"
)
$schema = Join-Path $repositoryRoot 'config\news_feature_schema_coarse.json'
$outputRoot = Join-Path $repositoryRoot 'outputs\news_provider_fulltext\v1_0'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Recovered Python runtime is missing: $python"
}
if (-not (Test-Path -LiteralPath $input)) {
    throw "Benchmark variant is missing: $input"
}

$overwriteArgument = @()
if ($Overwrite) {
    $overwriteArgument = @('--overwrite')
}

if ($Runner -eq 'flan-t5-xl') {
    $rawOutput = Join-Path $outputRoot "flan_t5_xl\$Variant.raw.jsonl"
    $finalOutput = Join-Path $outputRoot "flan_t5_xl\$Variant.jsonl"
    $sitePackages = Join-Path $repositoryRoot '.venv-flan-t5-xl\Lib\site-packages'
    $extractor = Join-Path $repositoryRoot 'scripts\run_flan_t5_xl_coarse.py'
    $calibrator = Join-Path $repositoryRoot 'scripts\experiment_flan_t5_xl_v1_1.py'
    $calibration = Join-Path $repositoryRoot 'experiments\flan_t5_xl\v1_1\calibration.json'

    # PyTorch/Transformers progress bars are written to stderr. PowerShell 5.1
    # exposes native stderr as non-terminating NativeCommandError records; do
    # not let those progress messages trip the script's fail-fast preference.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python $launcher `
        --site-packages $sitePackages `
        --script $extractor -- `
        --input $input `
        --output $rawOutput `
        --schema $schema `
        --model-id 'google/flan-t5-xl' `
        --revision '7d6315df2c2fb742f0f5b556879d730926ca9001' `
        --decoding 'order_averaged_letter_score' `
        --prompt-profile 'zero_shot' `
        --device 'cuda' `
        --precision 'float16' `
        --batch-size $FlanBatchSize `
        --local-files-only `
        @overwriteArgument
    $extractExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($extractExitCode -ne 0) {
        throw "FLAN-T5-XL extraction failed with exit code $extractExitCode"
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python $launcher `
        --site-packages $sitePackages `
        --script $calibrator -- `
        apply `
        --predictions $rawOutput `
        --schema $schema `
        --config $calibration `
        --output $finalOutput `
        @overwriteArgument
    $calibrationExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($calibrationExitCode -ne 0) {
        throw "FLAN-T5-XL calibration failed with exit code $calibrationExitCode"
    }
}
else {
    $finalOutput = Join-Path $outputRoot "llama_3_1\$Variant.jsonl"
    $sitePackages = Join-Path $repositoryRoot '.venv-llama31\Lib\site-packages'
    $extractor = Join-Path $repositoryRoot 'scripts\extract_llama_3_1_coarse.py'

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python $launcher `
        --site-packages $sitePackages `
        --script $extractor -- `
        --input $input `
        --output $finalOutput `
        @overwriteArgument
    $extractExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($extractExitCode -ne 0) {
        throw "Llama 3.1 extraction failed with exit code $extractExitCode"
    }
}

Write-Output "Completed $Runner / $Variant -> $finalOutput"
