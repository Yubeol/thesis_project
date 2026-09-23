import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agent.evidence import select_evidence_excerpt
from agent.finalizer.case_validator import detect_case_contradictions
from agent.finalizer.finalizer import _format_evidence
from agent.orchestrator import (
    _align_citations_to_sources,
    _build_sources,
    _has_news_citation,
    _has_recent_citable_news,
)
from agent.orchestrator import generate_paper
from agent.retrieval.hybrid import _rank_news_results, _rank_paper_passages


def test_comparison_result_near_end_of_chunk_reaches_finalizer():
    item = (
        "[PAPER 1]\nTitle: Indonesian fan protests\nEvidence:\n"
        + ("General background on concerts and fandom. " * 70)
        + "ELF did not receive collective support for its protest. "
        + "My Day's success depended on collective fan accounts and digital communities."
    )
    formatted = _format_evidence(
        [item],
        kind="PAPER",
        max_items=15,
        focus="DAY6 My Day and Super Junior ELF protest comparison",
    )

    assert "ELF did not receive collective support" in formatted
    assert "My Day's success depended" in formatted
    assert len(formatted) <= 1800
    assert formatted.startswith("[PAPER 1]\nTitle:")


def test_short_evidence_is_unchanged():
    item = "[PAPER 1]\nTitle: Test\nEvidence:\nShort original sentence."
    assert select_evidence_excerpt(item, max_chars=1800, focus="Test") == item


def test_relevant_second_chunk_not_hidden_behind_all_other_papers():
    items = [
        {"paper_id": 153, "chunk_id": 1, "similarity": 0.90},
        {"paper_id": 153, "chunk_id": 7, "similarity": 0.89},
    ] + [
        {"paper_id": index, "chunk_id": index, "similarity": 0.80}
        for index in range(1, 17)
    ]

    ranked = _rank_paper_passages(items)
    assert [item["chunk_id"] for item in ranked[:2]] == [1, 7]


def test_news_ranking_preserves_recency_adjusted_score():
    ranked = _rank_news_results([
        {"news_id": 1, "similarity": 0.80, "ranking_score": 0.80},
        {"news_id": 2, "similarity": 0.77, "ranking_score": 0.85},
    ])

    assert [item["news_id"] for item in ranked] == [2, 1]


def test_recent_citable_news_requires_content_url_and_recent_date():
    recent = datetime.now(timezone.utc) - timedelta(days=30)
    old = datetime.now(timezone.utc) - timedelta(days=366 * 4)

    assert _has_recent_citable_news({"news": [{
        "content": "A documented fandom campaign.",
        "url": "https://example.org/recent",
        "published_at": recent,
    }]}) is True
    assert _has_recent_citable_news({"news": [{
        "content": "An older report.",
        "url": "https://example.org/old",
        "published_at": old,
    }]}) is False
    assert _has_news_citation("사례 설명 [NEWS 2].") is True


def test_citation_numbers_match_deduplicated_reference_list():
    retrieval = {
        "papers": [
            {"content": "one", "title": "First", "source_url": "https://paper/1"},
            {"content": "two", "title": "Second", "source_url": "https://paper/2"},
            {"content": "more", "title": "First", "source_url": "https://paper/1"},
        ],
        "news": [
            {"content": "news", "title": "News", "url": "https://news/1"},
        ],
    }
    final = "Claim [PAPER 3]. Comparison [PAPER 2]. News [NEWS 1]."
    sources = _build_sources(retrieval, cited_text=final)
    aligned = _align_citations_to_sources(final, retrieval, sources)

    assert [source["title"] for source in sources] == ["Second", "First", "News"]
    assert aligned == "Claim [2]. Comparison [1]. News [3]."


def test_case_validator_limits_review_to_cited_papers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "contradiction_detected": True,
            "contradictions": ["The draft swaps My Day and ELF outcomes."]
        })))]
    )
    evidence = [
        "[PAPER 1]\nEvidence:\nMy Day achieved collective action; ELF did not.",
        "[PAPER 2]\nEvidence:\nUnrelated study.",
    ]

    with patch("agent.finalizer.case_validator.OpenAI") as client_type:
        client_type.return_value.chat.completions.create.return_value = response
        findings = detect_case_contradictions(
            title="My Day and ELF",
            draft="ELF achieved collective action [PAPER 1].",
            paper_evidence=evidence,
        )
        prompt = client_type.return_value.chat.completions.create.call_args.kwargs[
            "messages"
        ][1]["content"]

    assert findings == ["The draft swaps My Day and ELF outcomes."]
    assert "Unrelated study" not in prompt


def test_orchestrator_retries_reversed_case_and_aligns_citation():
    retrieval = {
        "papers": [{
            "paper_id": 153,
            "content": "My Day had collective support, whereas ELF did not.",
            "title": "Indonesian fandom protest",
            "source_url": "https://example.org/paper",
        }],
        "news": [],
    }
    analysis = SimpleNamespace(
        allowed=True,
        title="My Day와 ELF 비교",
        topic="팬덤 항의",
        research_question="어느 팬덤이 집단적 지원을 받았나?",
        instruction="학술적으로 작성",
        paper_queries=["fandom protest"],
        news_queries=[],
    )
    gaps = SimpleNamespace(model_dump=lambda: {})
    wrong = "서론\n비교.\n본론\nELF가 지원을 받았다 [PAPER 1].\n결론\n요약."
    corrected = "서론\n비교.\n본론\nMy Day가 지원을 받았다 [PAPER 1].\n결론\n요약."

    with (
        patch("agent.orchestrator.analyze_query", return_value=analysis),
        patch("agent.orchestrator.retrieve_hybrid", return_value=retrieval),
        patch("agent.orchestrator.generate_transformer_draft", return_value="draft"),
        patch("agent.orchestrator.analyze_gaps", return_value=gaps),
        patch("agent.orchestrator.run_adaptive_retrieval", return_value={
            "retrieval": retrieval, "performed": False
        }),
        patch("agent.orchestrator.finalize_draft", side_effect=[wrong, corrected]) as finalizer,
        patch("agent.orchestrator._limit_korean_final_draft", side_effect=lambda **kw: kw["final_text"]),
        patch("agent.orchestrator.detect_case_contradictions", side_effect=[
            ["The draft reverses the two fandoms."], []
        ]),
        patch("agent.orchestrator.extract_visuals", return_value=[]),
    ):
        result = generate_paper(topic="팬덤 항의", use_graph=False)

    assert finalizer.call_count == 2
    assert finalizer.call_args.kwargs["correction_notes"] == [
        "The draft reverses the two fandoms."
    ]
    assert "My Day가 지원을 받았다 [1]" in result["final"]
    assert result["sources"][0]["title"] == "Indonesian fandom protest"


def test_orchestrator_rechecks_omitted_recent_news_once():
    retrieval = {
        "papers": [{
            "paper_id": 1,
            "content": "Fan participation can create shared cultural meaning.",
            "title": "Fandom participation",
            "source_url": "https://example.org/paper",
            "similarity": 0.8,
        }],
        "news": [{
            "news_id": 1,
            "content": "In 2026, fans organized a documented online campaign.",
            "title_original": "Documented fan campaign",
            "published_at": datetime.now(timezone.utc),
            "source": "Example News",
            "url": "https://example.org/news",
            "similarity": 0.8,
            "ranking_score": 0.88,
        }],
        "debug": {},
    }
    analysis = SimpleNamespace(
        allowed=True,
        title="팬덤 참여와 홍보 노동",
        topic="팬덤의 온라인 참여",
        research_question="팬 활동은 어떤 의미를 갖는가?",
        instruction="학술적으로 작성",
        paper_queries=["fandom participation"],
        news_queries=["recent fandom campaign"],
    )
    gaps = SimpleNamespace(model_dump=lambda: {})
    without_news = (
        "서론\n문제 제기.\n본론\n문화적 참여 사례 [PAPER 1].\n결론\n요약."
    )
    with_news = (
        "서론\n문제 제기.\n본론\n문화적 참여 [PAPER 1]. "
        "2026년 팬 캠페인이 보도되었다 [NEWS 1].\n결론\n요약."
    )

    with (
        patch("agent.orchestrator.analyze_query", return_value=analysis),
        patch("agent.orchestrator.retrieve_hybrid", return_value=retrieval),
        patch("agent.orchestrator.generate_transformer_draft", return_value="draft"),
        patch("agent.orchestrator.analyze_gaps", return_value=gaps),
        patch("agent.orchestrator.run_adaptive_retrieval", return_value={
            "retrieval": retrieval, "performed": False
        }),
        patch(
            "agent.orchestrator.finalize_draft",
            side_effect=[without_news, with_news],
        ) as finalizer,
        patch(
            "agent.orchestrator._limit_korean_final_draft",
            side_effect=lambda **kw: kw["final_text"],
        ),
        patch("agent.orchestrator.detect_case_contradictions", return_value=[]),
        patch("agent.orchestrator.extract_visuals", return_value=[]),
    ):
        result = generate_paper(topic="팬덤 참여", use_graph=False)

    assert finalizer.call_count == 2
    assert finalizer.call_args.kwargs["news_case_retry"] is True
    assert "팬 캠페인이 보도되었다 [2]" in result["final"]
    assert [source["type"] for source in result["sources"]] == ["paper", "news"]
