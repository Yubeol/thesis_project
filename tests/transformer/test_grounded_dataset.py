import inspect
import threading
from unittest.mock import Mock, patch

import pytest

from transformer.dataset.build_grounded_dataset import (
    cache_key,
    quality_check,
    teacher_input,
    teacher_messages,
)
from transformer.inference.generate import DraftGenerator, SectionAdapterDraftGenerator, generate_draft


class TinyTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split()) + 1))}


def sample():
    return {
        "paper_id": 1,
        "target": "SECRET ORIGINAL TARGET NEVER SENT TO TEACHER",
        "input": {
            "title": "K-pop and online fandom",
            "topic": "online fandom",
            "research_question": "How does K-pop fandom use online communities?",
            "paper_evidence": ["[PAPER 1]\nTitle: K-pop and online fandom\nEvidence: Online fan communities share translated material and coordinate discussion about K-pop releases."],
            "news_evidence": ["[NO_NEWS_EVIDENCE]"],
            "instruction": "Write an academic section.",
            "section": "Body",
        },
    }


def test_teacher_prompt_never_sees_original_target():
    row = sample()
    payload = teacher_input(row)
    messages = teacher_messages(row)
    assert "target" not in payload
    assert "SECRET ORIGINAL TARGET" not in str(messages)
    assert "Requested Section: Body" in messages[1]["content"]
    assert "Paper Evidence:" in messages[1]["content"]


def test_cache_key_changes_with_evidence_not_original_target():
    row = sample()
    first = cache_key(row, "gpt-4o-mini")
    row["target"] = "a different original target"
    assert cache_key(row, "gpt-4o-mini") == first
    row["input"]["paper_evidence"][0] += " Changed evidence."
    assert cache_key(row, "gpt-4o-mini") != first


def test_quality_rejects_unsupported_numbers_and_names():
    row = sample()
    target = (
        "Online fan communities share translated material and coordinate discussion "
        "about K-pop releases. Harvard University reported that 73% of fans use "
        "these communities, although the evidence only describes their shared "
        "material and coordinated discussion. The research therefore identifies "
        "online communities as a place where fans exchange information and "
        "participate in discussion around releases."
    )
    result = quality_check(row, target, TinyTokenizer())
    assert "unsupported_number" in result["rejection_reason"]
    assert "unsupported_entity" in result["rejection_reason"]
    assert not result["grounding_pass"]


def test_quality_rejects_heading():
    row = sample()
    result = quality_check(row, "Body:\nOnline fans discuss K-pop.", TinyTokenizer())
    assert "heading" in result["rejection_reason"]


def test_public_generate_draft_contract_and_shared_section_routing():
    parameters = inspect.signature(generate_draft).parameters
    assert list(parameters)[:6] == [
        "title", "topic", "research_question", "paper_evidence", "news_evidence", "instruction"
    ]
    generator = object.__new__(DraftGenerator)

    def fake_section(**kwargs):
        return kwargs["section"], {"rejected": False}

    with patch.object(generator, "_generate_section_with_diagnostics", side_effect=fake_section) as called:
        result = generator.generate(title="Example", paper_evidence=["Evidence."])
    assert [call.kwargs["section"] for call in called.call_args_list] == ["Introduction", "Body", "Conclusion"]
    assert result == "Introduction:\nIntroduction\n\nBody:\nBody\n\nConclusion:\nConclusion"


def test_bad_model_path_fails_without_fallback(tmp_path):
    with pytest.raises(ValueError, match="Saved model not found"):
        DraftGenerator(tmp_path / "not-a-model", device="cpu")


def test_section_adapter_routes_before_raw_generation():
    generator = object.__new__(SectionAdapterDraftGenerator)
    generator.lock = threading.RLock()
    generator.model = Mock()
    with patch.object(DraftGenerator, "_generate_section_with_diagnostics", return_value=("raw", {"rejected": False})) as raw:
        assert generator._generate_section_with_diagnostics(section="Conclusion") == ("raw", {"rejected": False})
    generator.model.set_adapter.assert_called_once_with("Conclusion")
    raw.assert_called_once_with(section="Conclusion")
