param(
    [string]$VenvPath = "ouster-env",
    [string]$Device,
    [int]$Epochs = 100,
    [int]$Batch = 16,
    [int]$Imgsz = 640,
    [string]$Name = "mot17_baseline"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$activateScript = Join-Path $projectRoot "scripts\\activate_env.ps1"

. $activateScript -VenvPath $VenvPath

$cmd = @(
    "python",
    "train_baseline.py",
    "--epochs", $Epochs,
    "--batch", $Batch,
    "--imgsz", $Imgsz,
    "--name", $Name
)

if ($Device) {
    $cmd += @("--device", $Device)
}

Write-Host "Running: $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length - 1)]

