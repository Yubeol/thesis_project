from unittest.mock import patch

from agent.orchestrator import generate_paper, run_agent_pipeline
from agent.schemas import GapAnalysis, QueryAnalysis


TITLE_KO = "K-pop의 글로벌 확산에서 TikTok이 미친 영향"


MOCK_RETRIEVAL = {
    "papers": [
        {
            "chunk_id": 1,
            "paper_id": 7,
            "title": "Digital K-pop Fandom",
            "authors": ["Kim"],
            "published_year": 2024,
            "source": "Journal",
            "doi": "10.1000/example",
            "similarity": 0.81,
            "content": (
                "Social media platforms support transnational participation "
                "among global K-pop fandoms."
            ),
        }
    ],
    "news": [],
    "debug": {"paper_count": 1, "news_count": 0},
}


def allowed_analysis() -> QueryAnalysis:
    return QueryAnalysis(
        allowed=True,
        title=TITLE_KO,
        topic=TITLE_KO,
        research_question="TikTok은 K-pop 확산에 어떤 영향을 미쳤는가?",
        instruction="학술적인 문체로 작성",
        paper_queries=["TikTok K-pop global diffusion"],
        news_queries=["TikTok K-pop global diffusion"],
    )


def test_generate_paper_completed():
    gaps = GapAnalysis(
        needs_additional_retrieval=False,
        gaps=[],
    )

    with (
        patch("agent.orchestrator.analyze_query", return_value=allowed_analysis()),
        patch("agent.orchestrator.retrieve_hybrid", return_value=MOCK_RETRIEVAL),
        patch(
            "agent.orchestrator.generate_transformer_draft",
            return_value="Introduction:\nGrounded draft.",
        ) as transformer,
        patch("agent.orchestrator.analyze_gaps", return_value=gaps),
        patch(
            "agent.orchestrator.run_adaptive_retrieval",
            return_value={"performed": False, "retrieval": MOCK_RETRIEVAL},
        ),
        patch(
            "agent.orchestrator.finalize_draft",
            return_value="서론\n근거 기반 최종 초안.",
        ),
    ):
        result = generate_paper(topic=TITLE_KO)

    assert result["allowed"] is True
    assert result["final"] == "서론\n근거 기반 최종 초안."
    assert result["evidence_count"]["initial_papers"] == 1
    assert len(transformer.call_args.kwargs["paper_evidence"]) == 1


def test_run_agent_pipeline_rejects_out_of_scope_topic():
    rejected = QueryAnalysis(
        allowed=False,
        rejection_reason="지원하지 않는 주제입니다.",
    )

    with patch("agent.orchestrator.analyze_query", return_value=rejected):
        result = run_agent_pipeline(title_ko="오늘 날씨")

    assert result["status"] == "abstained"
    assert result["transformer_draft"] is None
    assert result["final_text"] is None
