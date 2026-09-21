import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from transformer.configs.config import TrainingConfig
from transformer.inference import generate_draft
from transformer.inference.generate import ABSTENTION, format_draft
from transformer.preprocessing.prompts import make_input, parse_sections
from transformer.training.train import validate_receipt


def test_full_training_requires_matching_receipt(tmp_path):
    signature = {"dataset": "snapshot-one", "config": {"batch_size": 1}}
    path = tmp_path / "dry_run_success.json"
    with pytest.raises(ValueError, match="requires"):
        validate_receipt(path, signature)
    path.write_text(json.dumps({"status": "passed", "signature": signature}))
    validate_receipt(path, signature)
    with pytest.raises(ValueError, match="does not match"):
        validate_receipt(path, {**signature, "dataset": "snapshot-two"})


def test_grounding_and_output_format():
    value = make_input("Example", paper_evidence=["Participants described shared activities."])
    assert value["research_question"].endswith("?")
    with pytest.raises(ValueError, match="At least one"):
        make_input("Example")
    text, diagnostics = format_draft("Introduction: A community. Body: 900 people participated. Conclusion: See Smith (2025).", "A community of 20 people.")
    assert diagnostics["rejected_sections"] == ["Body", "Conclusion"]
    assert parse_sections(text, strict=True)["Body"] == ABSTENTION
    text, diagnostics = format_draft("A short body without headings.", "A short body.")
    assert diagnostics["missing_sections"] == ["Introduction", "Conclusion"]
    assert set(parse_sections(text, strict=True)) == {"Introduction", "Body", "Conclusion"}
    text, diagnostics = format_draft("Body: The report was published by Imaginary Research Institute.", "Participants shared music.")
    assert diagnostics["rejected_sections"] == ["Body"]
    assert "Imaginary" not in text


def test_legacy_agent_call_and_new_signature():
    with patch("transformer.inference.generate._generator") as factory:
        factory.return_value.generate.return_value = "Draft"
        assert generate_draft(title="Example", topic="Fandom", evidence="Legacy combined RAG evidence.") == "Draft"
        assert factory.return_value.generate.call_args.kwargs["paper_evidence"] == ["Legacy combined RAG evidence."]
        generate_draft("Example", "Fandom", "What changed?", ["Paper evidence."], ["News evidence."], "Write a draft.")
        assert factory.return_value.generate.call_args.kwargs["news_evidence"] == ["News evidence."]


def test_conservative_defaults_and_invalid_parameters():
    config = TrainingConfig()
    assert config.batch_size == 1 and not config.fp16
    assert config.max_input_length <= 512
    assert replace(config, epochs=1).fingerprint_fields() == config.fingerprint_fields()
    for modified in (replace(config, learning_rate=float("nan")), replace(config, batch_size=0), replace(config, max_input_length=10)):
        with pytest.raises(ValueError):
            modified.validate()


def test_gpu_request_cannot_silently_fallback():
    torch = pytest.importorskip("torch")
    from transformer.training.runtime import check_device
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
            check_device("cuda")


def test_windows_workflow_contract():
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/train-transformer.yml").read_text(
            encoding="utf-8"
        )
    )
    train = workflow["jobs"]["train"]
    assert train["runs-on"] == ["self-hosted", "Windows", "X64", "gpu-train"]
    assert train["defaults"]["run"]["shell"] == "powershell"
    commands = "\n".join(step.get("run", "") for step in train["steps"])
    assert "--device cuda" in commands and "--dry-run-receipt" in commands
    assert "POSTGRES_PASSWORD" not in json.dumps(train)
