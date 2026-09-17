param(
    [string]$VenvPath = '.venv-transformer',
    [string]$PythonCommand = 'python'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

& $PythonCommand --version
if ($LASTEXITCODE -ne 0) { throw 'Python is not available to the Windows Runner service account.' }
& $PythonCommand -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for the tested Pascal configuration'"
if ($LASTEXITCODE -ne 0) { throw 'Select Python 3.12 before running this setup script.' }
& nvidia-smi
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed. Check the NVIDIA driver and service account PATH.' }

$pythonPath = Join-Path $VenvPath 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & $PythonCommand -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $pythonPath -c "import sys; assert sys.version_info[:2] == (3, 12), 'Existing training environment uses a different Python version'"
if ($LASTEXITCODE -ne 0) { throw 'Choose a new VenvPath for Python 3.12; the existing environment was preserved.' }
& $pythonPath -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'Pascal CUDA PyTorch installation failed. Inspect Python version, driver and wheel availability above.' }
}
# Preserve an existing torch wheel and test real kernels before installing other packages.
& $pythonPath -m transformer.training.runtime --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Existing torch is not CUDA-compatible. It was not removed or replaced; use a separate compatible environment.' }
& $pythonPath -m pip install -r transformer/configs/requirements-runtime.txt
if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
& $pythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Training dependency consistency check failed.' }
& $pythonPath -m transformer.training.runtime --device cuda
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed after dependency installation.' }
if ($env:GITHUB_ENV) {
    "TRAIN_PYTHON=$((Resolve-Path -LiteralPath $pythonPath).Path)" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
Write-Output "Training Python: $pythonPath"
