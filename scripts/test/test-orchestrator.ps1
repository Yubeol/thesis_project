param(
    [string]$Topic = "K-pop의 글로벌 확산에서 TikTok이 미친 영향",
    [string]$Instruction = "학술적인 문체로 작성"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = (
    Resolve-Path (
        Join-Path $PSScriptRoot "..\.."
    )
).Path


# -------------------------------------------------
# Python virtual environment
# -------------------------------------------------

$pythonCandidates = @(
    (Join-Path $projectRoot ".venv-transformer\Scripts\python.exe"),
    (Join-Path $projectRoot ".venv\Scripts\python.exe")
)

$python = $pythonCandidates |
    Where-Object {
        Test-Path -LiteralPath $_
    } |
    Select-Object -First 1

if (-not $python) {
    $pythonCommand = Get-Command python `
        -ErrorAction SilentlyContinue

    if ($pythonCommand) {
        $python = $pythonCommand.Source
    }
}

if (-not $python) {
    throw "Python virtual environment not found."
}

Write-Host "Python:" $python


# -------------------------------------------------
# Transformer model path
# -------------------------------------------------

if (-not $env:TRANSFORMER_MODEL_PATH) {

    $defaultModelPath = Join-Path `
        $projectRoot `
        "transformer\models\transformer-full\model"

    $modelFile = Join-Path `
        $defaultModelPath `
        "model.safetensors"

    if (Test-Path -LiteralPath $modelFile) {
        $env:TRANSFORMER_MODEL_PATH = (
            $defaultModelPath
        )
    }
}

if (-not $env:TRANSFORMER_MODEL_PATH) {
    throw (
        "TRANSFORMER_MODEL_PATH is not configured. " +
        "Set the environment variable or extract the model artifact."
    )
}

Write-Host (
    "Transformer model:"
) $env:TRANSFORMER_MODEL_PATH


# -------------------------------------------------
# Orchestrator E2E
# -------------------------------------------------

Push-Location $projectRoot

# -------------------------------------------------
# Orchestrator E2E
# -------------------------------------------------

Push-Location $projectRoot

try {

    $pythonCode = @'
import sys

from agent.orchestrator import generate_paper

topic = sys.argv[1]
instruction = sys.argv[2]

result = generate_paper(
    topic=topic,
    instruction=instruction,
)

print("ALLOWED:", result["allowed"])
print("TITLE:", result["title"])
print(
    "ADAPTIVE:",
    result["adaptive_retrieval_performed"],
)
print(
    "EVIDENCE:",
    result["evidence_count"],
)

print("=== TRANSFORMER ===")
print(result["draft"])

print("=== FINAL ===")
print(result["final"])
'@

    $pythonCode | & $python - $Topic $Instruction

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Orchestrator E2E test failed " +
            "with exit code $LASTEXITCODE"
        )
    }
}
finally {
    Pop-Location
}