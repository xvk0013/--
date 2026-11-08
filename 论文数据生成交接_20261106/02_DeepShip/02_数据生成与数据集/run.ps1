param(
    [ValidateSet('plan','smoke','full')][string]$Mode = 'smoke',
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$PythonExe = 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe',
    [string]$OutputDirectory = '',
    [switch]$ComponentsOnly,
    [switch]$MixOnly,
    [switch]$OverwritePlan,
    [switch]$OverwriteComponents
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonExe)) { $PythonExe = (Get-Command python -ErrorAction Stop).Source }
$env:PYTHONIOENCODING = 'utf-8'
$taskArguments = @('-u', (Join-Path $PSScriptRoot 'run.py'), '--config', $Config, '--mode', $Mode)
if ($OutputDirectory) { $taskArguments += @('--output', $OutputDirectory) }
if ($ComponentsOnly) { $taskArguments += '--components-only' }
if ($MixOnly) { $taskArguments += '--mix-only' }
if ($OverwritePlan) { $taskArguments += '--overwrite-plan' }
if ($OverwriteComponents) { $taskArguments += '--overwrite-components' }
& $PythonExe @taskArguments
if ($LASTEXITCODE -ne 0) { throw "DeepShip generation exited with code $LASTEXITCODE" }
