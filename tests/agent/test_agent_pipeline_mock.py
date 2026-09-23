from types import SimpleNamespace
from unittest.mock import patch

from agent.orchestrator import (
    generate_paper,
    run_agent_pipeline,
)
from agent.schemas import GapAnalysis


TITLE_KO = (
    "팬 번역 활동이 K-pop 콘텐츠의 "
    "해외 확산에 미치는 역할"
)

TOPIC_KO = (
    "팬 번역과 자막 활동이 언어 장벽 완화와 "
    "K-pop 콘텐츠의 해외 확산에 미치는 역할"
)


# =========================================================
# Mock Retrieval
# =========================================================

MOCK_RETRIEVAL = {
    "papers": [
        {
            "title": "Fan Translation and K-pop",
            "source_url": (
                "https://example.com/paper-1"
            ),
            "doi": "10.0000/example",
            "evidence_chunks": [
                {
                    "section": "abstract",
                    "content": (
                        "Fan translation can support "
                        "access to cross-border "
                        "media content."
                    ),
                }
            ],
        }
    ],

    "news": [
        {
            "title_original": (
                "K-pop 팬 번역 관련 기사"
            ),
            "url": (
                "https://example.com/news-1"
            ),
            "content": (
                "팬 번역 활동과 "
                "디지털 플랫폼에 관한 "
                "기사 내용."
            ),
        }
    ],

    "debug": {
        "mock": True,
    },
}


# =========================================================
# Query Analyzer Mock
# =========================================================

def allowed_analysis():
    """
    정상적으로 논문 생성을 허용한
    Query Analyzer 결과 Mock.
    """

    return SimpleNamespace(
        allowed=True,

        rejection_reason=None,

        title=TITLE_KO,

        topic=TOPIC_KO,

        research_question=(
            "팬 번역 활동은 "
            "K-pop 콘텐츠의 해외 확산에 "
            "어떤 역할을 하는가?"
        ),

        instruction=(
            "학술적인 문체로 작성"
        ),

        paper_queries=[
            (
                "K-pop fan translation "
                "global diffusion"
            ),
        ],

        news_queries=[
            (
                "K-pop fan translation "
                "overseas diffusion"
            ),
        ],
    )


def rejected_analysis():
    """
    Agent가 지원하지 않는 주제를
    거절한 상황의 Mock.
    """

    return SimpleNamespace(
        allowed=False,

        rejection_reason=(
            "지원 범위를 벗어난 주제입니다."
        ),

        title="범위 밖 주제",

        topic="범위 밖 주제",

        research_question="",

        instruction="",

        paper_queries=[],

        news_queries=[],
    )


# =========================================================
# 정상 논문 생성 Pipeline 테스트
# =========================================================

def test_generate_paper_completed():
    """
    전체 Agent 파이프라인이 정상적으로

    Query Analyzer
    → RAG
    → Transformer
    → Gap Analyzer
    → Adaptive RAG
    → Finalizer
    → 4500~4600자 Limiter

    순서로 실행되는지 확인한다.

    실제 API / DB / GPU는 사용하지 않는다.
    """

    gaps = GapAnalysis(
        needs_additional_retrieval=False,
        gaps=[],
    )

    # -----------------------------------------
    # Finalizer가 만든 결과
    # -----------------------------------------

    finalizer_output = (
        "서론\n"
        "근거 기반 서론 초안.\n\n"

        "본론\n"
        "검색된 근거를 바탕으로 "
        "구성한 본론 초안.\n\n"

        "결론\n"
        "검색 근거를 종합한 "
        "결론 초안."
    )

    # -----------------------------------------
    # output_limiter가 최종적으로
    # 반환했다고 가정하는 결과
    # -----------------------------------------

    limited_output = (
        "서론\n"
        "글자 수 보정이 적용된 "
        "최종 서론.\n\n"

        "본론\n"
        "글자 수 보정이 적용된 "
        "최종 본론.\n\n"

        "결론\n"
        "글자 수 보정이 적용된 "
        "최종 결론."
    )

    with (
        # -------------------------------------
        # LLM1 Query Analyzer
        # -------------------------------------

        patch(
            "agent.orchestrator.analyze_query",
            return_value=allowed_analysis(),
        ) as analyzer,

        # -------------------------------------
        # Hybrid RAG
        # -------------------------------------

        patch(
            "agent.orchestrator.retrieve_hybrid",
            return_value=MOCK_RETRIEVAL,
        ) as retrieval,

        # -------------------------------------
        # Evidence Builder
        #
        # 실제 Evidence 변환도 이 테스트에서는
        # Mock 처리한다.
        # -------------------------------------

        patch(
            "agent.orchestrator.build_evidence_lists",
            return_value=(
                [
                    (
                        "[PAPER] "
                        "grounded paper evidence"
                    )
                ],
                [
                    (
                        "[NEWS] "
                        "grounded news evidence"
                    )
                ],
            ),
        ) as evidence_builder,

        # -------------------------------------
        # Transformer
        # -------------------------------------

        patch(
            (
                "agent.orchestrator."
                "generate_transformer_draft"
            ),
            return_value=(
                "Introduction:\n"
                "Grounded draft.\n\n"

                "Body:\n"
                "Grounded body.\n\n"

                "Conclusion:\n"
                "Grounded conclusion."
            ),
        ) as transformer,

        # -------------------------------------
        # LLM2 Gap Analyzer
        # -------------------------------------

        patch(
            "agent.orchestrator.analyze_gaps",
            return_value=gaps,
        ) as gap_analyzer,

        # -------------------------------------
        # Adaptive RAG
        # -------------------------------------

        patch(
            (
                "agent.orchestrator."
                "run_adaptive_retrieval"
            ),
            return_value={
                "performed": False,
                "retrieval": MOCK_RETRIEVAL,
            },
        ) as adaptive_rag,

        # -------------------------------------
        # LLM3 Finalizer
        # -------------------------------------

        patch(
            "agent.orchestrator.finalize_draft",
            return_value=finalizer_output,
        ) as finalizer,

        # -------------------------------------
        # 한국어 4500~4600자 Limiter
        #
        # 실제 OpenAI를 호출하지 않고
        # 연결 여부만 확인한다.
        # -------------------------------------

        patch(
            (
                "agent.orchestrator."
                "_limit_korean_final_draft"
            ),
            return_value=limited_output,
        ) as limiter,
    ):

        result = generate_paper(
            topic=TITLE_KO,
        )

    # =====================================================
    # 결과 검증
    # =====================================================

    assert result["allowed"] is True

    assert (
        result["rejection_reason"]
        is None
    )

    assert (
        result["title"]
        == TITLE_KO
    )

    assert (
        result["topic"]
        == TOPIC_KO
    )

    # Transformer 결과 확인
    assert result["draft"].startswith(
        "Introduction:"
    )

    # 중요한 부분:
    # Finalizer 원본이 아니라
    # Limiter를 통과한 결과여야 한다.
    assert (
        result["final"]
        == limited_output
    )

    # Adaptive RAG 추가 검색 없음
    assert (
        result[
            "adaptive_retrieval_performed"
        ]
        is False
    )

    # =====================================================
    # Evidence 개수 확인
    # =====================================================

    assert result["evidence_count"] == {
        "initial_papers": 1,
        "initial_news": 1,
        "final_papers": 1,
        "final_news": 1,
    }

    # =====================================================
    # Retrieval Debug 확인
    # =====================================================

    assert result["retrieval_debug"] == {
        "mock": True,
    }

    # =====================================================
    # Frontend에 전달할 출처 확인
    # =====================================================

    assert len(
        result["sources"]
    ) == 2

    assert result["sources"][0] == {
        "type": "paper",
        "title": (
            "Fan Translation and K-pop"
        ),
        "url": (
            "https://example.com/paper-1"
        ),
    }

    assert result["sources"][1] == {
        "type": "news",
        "title": (
            "K-pop 팬 번역 관련 기사"
        ),
        "url": (
            "https://example.com/news-1"
        ),
    }

    # =====================================================
    # 각 단계 호출 여부 확인
    # =====================================================

    analyzer.assert_called_once()

    retrieval.assert_called_once()

    # build_evidence_lists는
    #
    # 1. 초기 전체 Evidence
    # 2. Transformer용 제한 Evidence
    # 3. Adaptive RAG 이후 최종 Evidence
    #
    # 총 3회 호출된다.
    assert (
        evidence_builder.call_count
        == 3
    )

    transformer.assert_called_once()

    gap_analyzer.assert_called_once()

    adaptive_rag.assert_called_once()

    finalizer.assert_called_once()

    # 새로 추가한 글자 수 Limiter가
    # 반드시 호출되어야 한다.
    limiter.assert_called_once()

    # Limiter가 Finalizer 결과를
    # 정확히 받았는지도 확인한다.
    limiter.assert_called_once_with(
        title=TITLE_KO,

        topic=TOPIC_KO,

        research_question=(
            "팬 번역 활동은 "
            "K-pop 콘텐츠의 해외 확산에 "
            "어떤 역할을 하는가?"
        ),

        final_text=finalizer_output,
    )


# =========================================================
# 지원 범위 밖 주제 거절 테스트
# =========================================================

def test_generate_paper_rejected_stops_pipeline():
    """
    Query Analyzer가 주제를 거절하면

    RAG
    Transformer
    Finalizer
    Output Limiter

    등이 실행되지 않는지 확인한다.
    """

    with (
        patch(
            "agent.orchestrator.analyze_query",
            return_value=rejected_analysis(),
        ) as analyzer,

        patch(
            "agent.orchestrator.retrieve_hybrid",
        ) as retrieval,

        patch(
            (
                "agent.orchestrator."
                "generate_transformer_draft"
            ),
        ) as transformer,

        patch(
            "agent.orchestrator.finalize_draft",
        ) as finalizer,

        patch(
            (
                "agent.orchestrator."
                "_limit_korean_final_draft"
            ),
        ) as limiter,
    ):

        result = generate_paper(
            topic="범위 밖 주제",
        )

    # =====================================================
    # 반환 결과 확인
    # =====================================================

    assert (
        result["allowed"]
        is False
    )

    assert (
        result["rejection_reason"]
        == "지원 범위를 벗어난 주제입니다."
    )

    assert (
        result["sources"]
        == []
    )

    assert (
        result["draft"]
        is None
    )

    assert (
        result["final"]
        is None
    )

    assert (
        result["gap_analysis"]
        is None
    )

    assert (
        result[
            "adaptive_retrieval_performed"
        ]
        is False
    )

    assert result["evidence_count"] == {
        "initial_papers": 0,
        "initial_news": 0,
        "final_papers": 0,
        "final_news": 0,
    }

    assert (
        result["retrieval_debug"]
        == {}
    )

    # =====================================================
    # 거절됐으므로 뒤 단계는 실행되면 안 된다.
    # =====================================================

    analyzer.assert_called_once()

    retrieval.assert_not_called()

    transformer.assert_not_called()

    finalizer.assert_not_called()

    limiter.assert_not_called()


# =========================================================
# 기존 run_agent_pipeline 호환성 테스트
# =========================================================

def test_run_agent_pipeline_compatibility_fields():
    """
    예전 Backend 코드가 사용하는

    status
    transformer_draft
    final_text
    message

    필드가 계속 유지되는지 확인한다.
    """

    completed_result = {
        "allowed": True,

        "rejection_reason": None,

        "title": TITLE_KO,

        "topic": TOPIC_KO,

        "research_question": "",

        "draft": (
            "Transformer draft"
        ),

        "final": (
            "서론\n"
            "최종 서론\n\n"

            "본론\n"
            "최종 본론\n\n"

            "결론\n"
            "최종 결론"
        ),

        "sources": [],

        "gap_analysis": {
            "needs_additional_retrieval":
                False,
            "gaps": [],
        },

        "adaptive_retrieval_performed":
            False,

        "evidence_count": {
            "initial_papers": 1,
            "initial_news": 1,
            "final_papers": 1,
            "final_news": 1,
        },

        "retrieval_debug": {},
    }

    with patch(
        "agent.orchestrator.generate_paper",
        return_value=completed_result,
    ) as generator:

        result = run_agent_pipeline(
            title_ko=TITLE_KO,
            topic_ko=TOPIC_KO,
        )

    assert (
        result["status"]
        == "completed"
    )

    assert (
        result["transformer_draft"]
        == "Transformer draft"
    )

    assert result["final_text"] == (
        "서론\n"
        "최종 서론\n\n"

        "본론\n"
        "최종 본론\n\n"

        "결론\n"
        "최종 결론"
    )

    assert (
        result["message"]
        is None
    )

    generator.assert_called_once_with(
        title=TITLE_KO,
        topic=TOPIC_KO,
    )