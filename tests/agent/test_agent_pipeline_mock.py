from unittest.mock import patch

from agent.orchestrator import run_agent_pipeline


TITLE_KO = (
    "소셜미디어 기반 글로벌 팬덤 활동이 "
    "K-POP의 세계적 확산에 미치는 영향"
)


MOCK_RETRIEVAL_RESULT = {
    "input": {
        "title_ko": TITLE_KO,
        "topic_ko": None,
    },
    "translated": {
        "title_en": (
            "The Impact of Social Media-Based Global "
            "Fandom Activities on the Global Diffusion of K-Pop"
        ),
        "topic_en": None,
        "query_en": (
            "The Impact of Social Media-Based Global "
            "Fandom Activities on the Global Diffusion of K-Pop"
        ),
    },
    "retrieval": {
        "papers": [
            {
                "paper_id": 1,
                "title": "Global K-pop Fandom",
                "authors": "Kim",
                "published_year": 2024,
                "retrieval_sources": [
                    "vector",
                    "graph",
                ],
                "evidence_chunks": [
                    {
                        "section": "body",
                        "content": (
                            "Social media platforms support "
                            "transnational participation among "
                            "global K-pop fandoms."
                        ),
                        "similarity": 0.81,
                    }
                ],
            }
        ],
        "news": [],
    },
    "decision": {
        "action": "generate",
        "message": None,
        "validation": {
            "can_generate": True,
            "status": "sufficient",
            "reasons": [],
        },
    },
}


def test_agent_pipeline_completed():
    """
    근거가 충분한 경우:
    Retrieval → Transformer → LLM → Korean
    → 최종 출력까지 진행되는지 확인한다.
    """

    with (
        patch(
            "agent.orchestrator.run_retrieval_pipeline",
            return_value=MOCK_RETRIEVAL_RESULT,
        ),
        patch(
            "agent.orchestrator.generate_transformer_draft",
            return_value=(
                "Transformer generated academic draft."
            ),
        ),
        patch(
            "agent.orchestrator.finalize_english_draft",
            return_value={
                "introduction": (
                    "Social media plays an important role "
                    "in the global diffusion of K-pop."
                ),
                "body": (
                    "Global fandom activities facilitate "
                    "content sharing and participation."
                ),
                "conclusion": (
                    "These activities contribute to the "
                    "international diffusion of K-pop."
                ),
            },
        ),
        patch(
            "agent.orchestrator.translate_draft_to_korean",
            return_value={
                "introduction": (
                    "소셜미디어는 K-POP의 세계적 확산에서 "
                    "중요한 역할을 한다."
                ),
                "body": (
                    "글로벌 팬덤 활동은 콘텐츠 공유와 "
                    "참여를 촉진한다."
                ),
                "conclusion": (
                    "이러한 활동은 K-POP의 국제적 확산에 "
                    "기여한다."
                ),
            },
        ),
    ):
        result = run_agent_pipeline(
            title_ko=TITLE_KO
        )

    assert result["status"] == "completed"

    assert result["draft_ko"]["title"] == TITLE_KO

    assert result["final_text"] is not None

    assert "서론" in result["final_text"]
    assert "본론" in result["final_text"]
    assert "결론" in result["final_text"]

    assert result["character_count"] <= 4500


def test_agent_pipeline_abstains():
    """
    근거가 부족한 경우:
    Transformer와 LLM을 호출하지 않고
    중단되는지 확인한다.
    """

    mock_abstain_result = {
        **MOCK_RETRIEVAL_RESULT,
        "decision": {
            "action": "abstain",
            "message": (
                "검색된 근거가 충분하지 않아 "
                "논문 초안을 생성하지 않았습니다."
            ),
            "validation": {
                "can_generate": False,
                "status": "insufficient",
                "reasons": [
                    "근거 부족"
                ],
            },
        },
    }

    with patch(
        "agent.orchestrator.run_retrieval_pipeline",
        return_value=mock_abstain_result,
    ):
        result = run_agent_pipeline(
            title_ko=TITLE_KO
        )

    assert result["status"] == "abstained"

    assert result["transformer_draft"] is None
    assert result["draft_en"] is None
    assert result["draft_ko"] is None
    assert result["final_text"] is None