param([string]$Python = '', [switch]$Wavs)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $preferredPython = Join-Path $env:USERPROFILE 'anaconda3\envs\zpfenv1\python.exe'
    if (Test-Path -LiteralPath $preferredPython) { $Python = $preferredPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
Push-Location $PSScriptRoot
try {
    if ($Wavs) { & $Python -B -X utf8 generator/audit.py --wavs }
    else { & $Python -B -X utf8 generator/audit.py }
    if ($LASTEXITCODE -ne 0) { throw 'Dataset acceptance failed.' }
} finally { Pop-Location }
