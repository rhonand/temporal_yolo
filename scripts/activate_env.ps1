param(
    [string]$VenvPath = "ouster-env"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot $VenvPath
$activateScript = Join-Path $venvRoot "Scripts\\Activate.ps1"
$cfgPath = Join-Path $venvRoot "pyvenv.cfg"

if (-not (Test-Path $venvRoot)) {
    throw "Virtual environment not found: $venvRoot"
}

if (-not (Test-Path $activateScript)) {
    throw "Activation script not found: $activateScript"
}

if (-not (Test-Path $cfgPath)) {
    throw "pyvenv.cfg not found: $cfgPath"
}

$cfgText = Get-Content $cfgPath -Raw
if ($cfgText -notmatch "version = 3\.10") {
    Write-Warning "This environment is not marked as Python 3.10. Please double-check the interpreter."
}

. $activateScript

Write-Host "Activated environment: $venvRoot"
Write-Host "Python on PATH: $(Get-Command python | Select-Object -ExpandProperty Source)"
python --version

