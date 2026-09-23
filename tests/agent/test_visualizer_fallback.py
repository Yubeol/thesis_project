from unittest.mock import patch

from agent.visualizer.extractor import extract_visuals


SOURCES = [
    {
        "type": "paper",
        "title": "Qualitative fandom study",
        "url": "https://example.org/paper",
    }
]

RETRIEVAL = {
    "papers": [
        {
            "title": "Qualitative fandom study",
            "source_url": "https://example.org/paper",
            "content": (
                "Fans coordinated translation work through online communities. "
                "Participants described collective verification before publication. "
                "The study also reported limits caused by unequal platform access."
            ),
        }
    ],
    "news": [],
}


def test_visualizer_uses_grounded_table_when_llm_fails():
    with patch("agent.visualizer.extractor.OpenAI", side_effect=RuntimeError("offline")):
        visuals = extract_visuals(
            title="팬덤 번역 활동",
            topic="온라인 협업",
            research_question="팬들은 어떻게 협업하는가?",
            retrieval=RETRIEVAL,
            sources=SOURCES,
        )

    assert len(visuals) == 1
    assert visuals[0]["kind"] == "table"
    assert len(visuals[0]["rows"]) >= 2
    assert "Fans coordinated translation work" in str(visuals[0]["rows"])
    assert visuals[0]["source_index"] == 0


def test_visualizer_retries_table_only_result_to_prefer_chart():
    table = {
        "kind": "table",
        "title": "근거 비교",
        "columns": ["구분", "내용"],
        "rows": [["협업", "Fans coordinated translation work"], ["검증", "collective verification"]],
        "source_index": 0,
    }
    chart = {
        "kind": "bar",
        "title": "활동 비교",
        "labels": ["협업", "검증"],
        "series": [{"name": "빈도", "values": [2, 3]}],
        "unit": "건",
        "source_index": 0,
    }
    numeric_retrieval = {
        "papers": [{
            **RETRIEVAL["papers"][0],
            "content": RETRIEVAL["papers"][0]["content"] + " Activities numbered 2 and 3.",
        }],
        "news": [],
    }

    with (
        patch("agent.visualizer.extractor.OpenAI"),
        patch("agent.visualizer.extractor._request_visuals", side_effect=[[table], [chart]]) as request,
    ):
        visuals = extract_visuals(
            title="팬덤 활동 비교",
            topic="협업과 검증",
            research_question="두 활동은 어떻게 다른가?",
            retrieval=numeric_retrieval,
            sources=SOURCES,
        )

    assert request.call_count == 2
    assert visuals[0]["kind"] == "bar"
