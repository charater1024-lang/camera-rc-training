param([switch]$CpuOnly, [switch]$WithDetection, [switch]$WithRuntime)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$wheelIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu126' }
& $venvPython -m pip install 'torch==2.9.1' 'torchvision==0.24.1' --index-url $wheelIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
$extras = @()
if ($WithDetection) { $extras += 'detection' }
if ($WithRuntime) { $extras += 'runtime' }
$projectSpec = if ($extras.Count) { $projectRoot + '[' + ($extras -join ',') + ']' } else { $projectRoot }
& $venvPython -m pip install -e $projectSpec
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
& $venvPython -m camera_rc doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
