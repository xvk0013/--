param(
    [ValidateSet('plan','pilot','full')]
    [string]$Mode = 'pilot',
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$PythonExe = 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe',
    [string]$RunDirectory = '',
    [int]$Workers = 0,
    [int]$PilotRecordingsPerClass = 0,
    [switch]$Plots,
    [switch]$Overwrite
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
$env:PYTHONIOENCODING = 'utf-8'
$taskArguments = @((Join-Path $PSScriptRoot 'run.py'), '--config', $Config, '--mode', $Mode)
if ($RunDirectory) { $taskArguments += @('--run-dir', $RunDirectory) }
if ($Workers -gt 0) { $taskArguments += @('--workers', [string]$Workers) }
if ($PilotRecordingsPerClass -gt 0) { $taskArguments += @('--pilot-recordings-per-class', [string]$PilotRecordingsPerClass) }
if ($Plots) { $taskArguments += '--plots' }
if ($Overwrite) { $taskArguments += '--overwrite' }
& $PythonExe @taskArguments
if ($LASTEXITCODE -ne 0) { throw "DeepShip prescreen exited with code $LASTEXITCODE" }
