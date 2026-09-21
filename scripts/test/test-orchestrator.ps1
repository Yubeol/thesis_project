param(
    [string]$Topic = "K-pop의 글로벌 확산에서 TikTok이 미친 영향",
    [string]$Instruction = "학술적인 문체로 작성"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $projectRoot ".venv-transformer\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Transformer virtual environment not found: $python"
}

Push-Location $projectRoot

try {
    & $python -c @'
import sys

from agent.orchestrator import generate_paper

topic = sys.argv[1]
instruction = sys.argv[2]
result = generate_paper(topic=topic, instruction=instruction)

print("ALLOWED:", result["allowed"])
print("TITLE:", result["title"])
print("ADAPTIVE:", result["adaptive_retrieval_performed"])
print("EVIDENCE:", result["evidence_count"])
print("=== TRANSFORMER ===")
print(result["draft"])
print("=== FINAL ===")
print(result["final"])
'@ $Topic $Instruction

    if ($LASTEXITCODE -ne 0) {
        throw "Orchestrator E2E test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
