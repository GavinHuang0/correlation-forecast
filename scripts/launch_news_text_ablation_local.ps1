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

    [Parameter(Mandatory = $true)]
    [string]$LogPath,

    [int]$FlanBatchSize = 32,

    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
$worker = Join-Path $PSScriptRoot 'run_news_text_ablation_local.ps1'
$resolvedLogPath = [System.IO.Path]::GetFullPath($LogPath)
$logDirectory = Split-Path -Parent $resolvedLogPath
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

try {
    # The worker validates native exit codes explicitly. Keep PowerShell 5.1
    # from promoting Transformers progress output on stderr into a terminating
    # NativeCommandError in this outer redirection wrapper.
    $ErrorActionPreference = 'Continue'
    & $worker `
        -Runner $Runner `
        -Variant $Variant `
        -FlanBatchSize $FlanBatchSize `
        -Overwrite:$Overwrite *> $resolvedLogPath
    exit $LASTEXITCODE
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $resolvedLogPath -Encoding UTF8
    exit 1
}
