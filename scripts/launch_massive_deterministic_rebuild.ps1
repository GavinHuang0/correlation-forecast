param(
    [string]$LogPath = (
        'outputs\news_provider_fulltext\v1_0\runtime\' +
        'deterministic_rebuild.log'
    )
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repositoryRoot '.venv-training\Scripts\python.exe'
$builder = Join-Path $PSScriptRoot 'build_massive_deterministic_news_features.py'
$resolvedLogPath = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot $LogPath)
)
$logDirectory = Split-Path -Parent $resolvedLogPath
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

$arguments = @(
    $builder,
    '--raw-root',
    (Join-Path $repositoryRoot 'data\raw\massive\ordinary_news\free_v1'),
    '--raw-root',
    (
        Join-Path $repositoryRoot (
            'data\raw\massive\ordinary_news\free_v1_benchmarks'
        )
    ),
    '--universe',
    (Join-Path $repositoryRoot 'config\news_target_universe_30.json'),
    '--calendar',
    (
        Join-Path $repositoryRoot (
            'data\prices\alpaca\calendar\2016-01-01_2026-06-30.json'
        )
    ),
    '--output-dir',
    (
        Join-Path $repositoryRoot (
            'data\features\news_deterministic\massive_v1'
        )
    ),
    '--start',
    '2016-06-22',
    '--end',
    '2026-06-30',
    '--overwrite'
)

try {
    # Keep PowerShell 5.1 from turning native stderr into a terminating
    # NativeCommandError while preserving the complete builder log.
    $ErrorActionPreference = 'Continue'
    & $python @arguments *> $resolvedLogPath
    exit $LASTEXITCODE
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $resolvedLogPath -Encoding UTF8
    exit 1
}
