from types import SimpleNamespace
from unittest.mock import patch

from agent.draft_generator.grounded_fallback import (
    ABSTENTION,
    replace_abstained_draft,
)
from agent.grounding import unsupported_numeric_claims
from agent.finalizer import finalize_draft
from agent.schemas import GapAnalysis


def test_abstained_transformer_output_uses_only_retrieved_passages():
    evidence = [
        "[PAPER 1]\nTitle: Example\nEvidence:\n"
        "Digital platforms support participation across national borders. "
        "Fans share media and coordinate collective activities online."
    ]
    draft = "\n\n".join(
        f"{section}:\n{ABSTENTION}"
        for section in ("Introduction", "Body", "Conclusion")
    )

    result, used = replace_abstained_draft(
        draft,
        title="Digital fandom",
        topic="Online participation",
        research_question="How do fans participate?",
        paper_evidence=evidence,
        news_evidence=[],
    )

    assert used is True
    assert ABSTENTION not in result
    assert "Fans share media" in result
    assert "Introduction:" in result
    assert "Body:" in result
    assert "Conclusion:" in result


def test_unsupported_numeric_claims_are_detected():
    unsupported = unsupported_numeric_claims(
        "The platform reached 2,500 users in 2026, an increase of 18%.",
        title="Platform study",
        topic="Participation",
        research_question="What changed?",
        paper_evidence=["The study surveyed 2,500 users in 2025."],
        news_evidence=[],
    )

    assert unsupported == ["18%", "2026"]


def test_repeated_title_only_output_uses_fallback():
    draft = (
        "Introduction:\nDigital fandom participation\n\n"
        "Body:\nDigital fandom participation\n\n"
        "Conclusion:\nDigital fandom participation"
    )
    result, used = replace_abstained_draft(
        draft,
        title="Digital fandom participation",
        topic="Online participation",
        research_question="How do fans participate?",
        paper_evidence=[
            "Evidence: Digital platforms support participation across "
            "national borders. Fans coordinate collective activities online."
        ],
        news_evidence=[],
    )

    assert used is True
    assert "Digital platforms support participation" in result


def test_finalizer_rewrites_unsupported_number(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    first = "The platform increased participation by 99%."
    corrected = "The evidence indicates increased participation."

    with patch("agent.finalizer.finalizer.OpenAI") as client_type:
        client_type.return_value.chat.completions.create.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=first))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=corrected))]
            ),
        ]

        result = finalize_draft(
            title="Digital fandom",
            topic="Participation",
            research_question="What changed?",
            draft="Introduction: Evidence indicates participation.",
            gap_analysis=GapAnalysis(),
            paper_evidence=["Participants described increased engagement."],
            news_evidence=[],
        )

    assert result == corrected
    assert client_type.return_value.chat.completions.create.call_count == 2
