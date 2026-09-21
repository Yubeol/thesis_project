from typing import Any

from agent.query_analyzer import analyze_query
from agent.retrieval import retrieve_hybrid
from agent.evidence import build_evidence_lists
from agent.draft_generator import generate_transformer_draft
from agent.adaptive_rag import (
    analyze_gaps,
    run_adaptive_retrieval,
)
from agent.finalizer import finalize_draft


def run_retrieval_pipeline(
    title_ko: str,
    topic_ko: str | None = None,
    *,
    use_graph: bool = True,
    strict_graph: bool = False,
) -> dict[str, Any]:
    """
    Retrieval까지만 수행하는 호환용 파이프라인.

    LLM1 입력 분석
    → 1차 Hybrid RAG
    → Evidence 변환
    """

    topic = topic_ko or title_ko

    analysis = analyze_query(
        title=title_ko,
        topic=topic,
        research_question="",
        instruction="",
    )

    if not analysis.allowed:
        return {
            "allowed": False,
            "rejection_reason": analysis.rejection_reason,
            "analysis": analysis.model_dump(),
            "retrieval": None,
            "paper_evidence": [],
            "news_evidence": [],
        }

    retrieval = retrieve_hybrid(
        paper_queries=analysis.paper_queries,
        news_queries=analysis.news_queries,
        use_graph=use_graph,
        strict_graph=strict_graph,
    )

    paper_evidence, news_evidence = build_evidence_lists(
        retrieval
    )

    return {
        "allowed": True,
        "rejection_reason": None,
        "analysis": analysis.model_dump(),
        "retrieval": retrieval,
        "paper_evidence": paper_evidence,
        "news_evidence": news_evidence,
    }


def generate_paper(
    *,
    title: str = "",
    topic: str = "",
    research_question: str = "",
    instruction: str = "",
    use_graph: bool = True,
    strict_graph: bool = False,
) -> dict[str, Any]:
    """
    전체 논문 생성 파이프라인.

    LLM1
    → 1차 Hybrid RAG
    → Transformer 초안
    → LLM2 Gap Analyzer
    → 필요 시 2차 Hybrid RAG
    → LLM3 Finalizer
    """

    # 1. LLM1
    analysis = analyze_query(
        title=title,
        topic=topic,
        research_question=research_question,
        instruction=instruction,
    )

    # 범위 밖 주제는 즉시 종료
    if not analysis.allowed:
        return {
            "allowed": False,
            "rejection_reason": analysis.rejection_reason,
            "title": analysis.title,
            "topic": analysis.topic,
            "research_question": analysis.research_question,
            "draft": None,
            "final": None,
            "gap_analysis": None,
            "adaptive_retrieval_performed": False,
            "evidence_count": {
                "initial_papers": 0,
                "initial_news": 0,
                "final_papers": 0,
                "final_news": 0,
            },
            "retrieval_debug": {},
        }

    # 2. 1차 Hybrid RAG
    initial_retrieval = retrieve_hybrid(
        paper_queries=analysis.paper_queries,
        news_queries=analysis.news_queries,
        use_graph=use_graph,
        strict_graph=strict_graph,
    )

    paper_evidence, news_evidence = build_evidence_lists(
        initial_retrieval
    )

    # The local model has a 384-token input contract. Passing every result
    # from every expanded query leaves only a few tokens per evidence item.
    # Keep the complete lists for LLM2/LLM3, but give Transformer the most
    # relevant bounded subset.
    transformer_paper_evidence, transformer_news_evidence = (
        build_evidence_lists(
            initial_retrieval,
            max_chars_per_item=1200,
            max_papers=4,
            max_news=2,
        )
    )

    # 3. Transformer
    draft = generate_transformer_draft(
        title=analysis.title,
        topic=analysis.topic,
        research_question=analysis.research_question,
        paper_evidence=transformer_paper_evidence,
        news_evidence=transformer_news_evidence,
        instruction=analysis.instruction,
    )

    # 4. LLM2
    gap_analysis = analyze_gaps(
        title=analysis.title,
        topic=analysis.topic,
        research_question=analysis.research_question,
        draft=draft,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
    )

    # 5. 필요 시 2차 RAG
    adaptive_result = run_adaptive_retrieval(
        gap_analysis=gap_analysis,
        current_retrieval=initial_retrieval,
        use_graph=use_graph,
        strict_graph=strict_graph,
    )

    final_retrieval = adaptive_result["retrieval"]

    final_paper_evidence, final_news_evidence = (
        build_evidence_lists(
            final_retrieval
        )
    )

    # 6. LLM3
    final = finalize_draft(
        title=analysis.title,
        topic=analysis.topic,
        research_question=analysis.research_question,
        draft=draft,
        gap_analysis=gap_analysis,
        paper_evidence=final_paper_evidence,
        news_evidence=final_news_evidence,
    )

    # 7. 반환
    return {
        "allowed": True,
        "rejection_reason": None,

        "title": analysis.title,
        "topic": analysis.topic,
        "research_question": analysis.research_question,

        "draft": draft,
        "final": final,

        "gap_analysis": gap_analysis.model_dump(),

        "adaptive_retrieval_performed": (
            adaptive_result["performed"]
        ),

        "evidence_count": {
            "initial_papers": len(
                initial_retrieval.get("papers", [])
            ),
            "initial_news": len(
                initial_retrieval.get("news", [])
            ),
            "final_papers": len(
                final_retrieval.get("papers", [])
            ),
            "final_news": len(
                final_retrieval.get("news", [])
            ),
        },

        "retrieval_debug": final_retrieval.get(
            "debug",
            {},
        ),
    }


def run_agent_pipeline(
    title_ko: str,
    topic_ko: str | None = None,
) -> dict[str, Any]:
    """
    기존 호출 코드 호환용 진입점.
    내부적으로 새 generate_paper() 파이프라인을 사용한다.
    """

    result = generate_paper(
        title=title_ko,
        topic=topic_ko or title_ko,
    )

    # 기존 호출부에서 사용하던 키 일부 유지
    return {
        **result,
        "status": (
            "completed"
            if result["allowed"]
            else "abstained"
        ),
        "transformer_draft": result.get("draft"),
        "final_text": result.get("final"),
        "message": result.get("rejection_reason"),
    }
