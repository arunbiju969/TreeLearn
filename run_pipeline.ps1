# Run TreeLearn Segmentation Pipeline with CUDA
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$appDataPython = "$env:LOCALAPPDATA\TreeLearn_venv\Scripts\python.exe"
if (Test-Path $appDataPython) {
    $pythonExe = $appDataPython
} else {
    $pythonExe = "$scriptDir\.venv\Scripts\python.exe"
}

Write-Host "Running TreeLearn Segmentation Pipeline on GPU using $pythonExe..." -ForegroundColor Cyan
& $pythonExe "$scriptDir\tools\pipeline\pipeline.py" --config "$scriptDir\configs\pipeline\pipeline.yaml"
