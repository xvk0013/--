param([string]$Python = '', [switch]$RebuildComponents, [switch]$TripleOnly, [switch]$PlanOnly)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $preferredPython = Join-Path $env:USERPROFILE 'anaconda3\envs\zpfenv1\python.exe'
    if (Test-Path -LiteralPath $preferredPython) { $Python = $preferredPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
Push-Location $PSScriptRoot
try {
    $settings = Get-Content -LiteralPath 'config.json' -Raw | ConvertFrom-Json
    if ($TripleOnly -or $settings.triple.enabled) {
        if ($RebuildComponents) { throw 'Triple-only generation reuses existing components; do not combine with RebuildComponents.' }
        $arguments = @('-B','-X','utf8','generator/generate.py','--triple-only')
        if ($PlanOnly) { $arguments += '--plan' }
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) { throw 'Triple generation failed.' }
        return
    }
    if ($PlanOnly) { throw 'Use -PlanOnly together with -TripleOnly.' }
    $sourceRows = Get-Content -LiteralPath 'metadata/sources.json' -Raw | ConvertFrom-Json
    $recordRows = Get-Content -LiteralPath 'metadata/recordings.json' -Raw | ConvertFrom-Json
    $cachePaths = @($sourceRows | ForEach-Object { $_.component_path })
    $cachePaths += @($recordRows | Where-Object { $_.class_name -eq 'E' } | ForEach-Object { $_.background_path })
    $missing = @($cachePaths | Where-Object { -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $_)) })
    if ($RebuildComponents -or $missing.Count -gt 0) {
        & $Python -B -X utf8 generator/prepare_audit.py --raw-only
        if ($LASTEXITCODE -ne 0) { throw 'Raw recording verification failed.' }
        & matlab -batch 'prepare_components'
        if ($LASTEXITCODE -ne 0) { throw 'MATLAB component reconstruction failed.' }
        & $Python -B -X utf8 generator/prepare_audit.py
        if ($LASTEXITCODE -ne 0) { throw 'Rebuilt component verification failed.' }
    }
    & $Python -B -X utf8 generator/generate.py
    if ($LASTEXITCODE -ne 0) { throw 'Generation or data check failed.' }
} finally { Pop-Location }
