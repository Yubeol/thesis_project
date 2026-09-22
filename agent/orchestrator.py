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

def _build_sources(
    retrieval: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    최종 RAG 검색 결과에서
    Frontend에 노출할 근거 목록을 생성한다.
    """

    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # -------------------------
    # Papers
    # -------------------------

    for paper in retrieval.get("papers", []):
        title = str(
            paper.get("title") or ""
        ).strip()

        url = str(
            paper.get("source_url") or ""
        ).strip()

        # source_url이 없으면 DOI 링크 사용
        if not url:
            doi = str(
                paper.get("doi") or ""
            ).strip()

            if doi:
                doi = doi.removeprefix("doi:").strip()

                if doi.startswith(
                    ("http://", "https://")
                ):
                    url = doi
                else:
                    url = f"https://doi.org/{doi}"

        # 프론트에서 실제 출처로 보여줄 수 있는
        # 제목 + URL이 있는 항목만 포함
        if not title or not url:
            continue

        key = ("paper", url)

        if key in seen:
            continue

        seen.add(key)

        similarity = paper.get("similarity")

        score = (
            round(float(similarity), 4)
            if isinstance(similarity, (int, float))
            else None
        )

        sources.append(
            {
                "type": "paper",
                "title": title,
                "url": url,
                "score": score,
            }
        )

    # -------------------------
    # News
    # -------------------------

    for news in retrieval.get("news", []):
        title = str(
            news.get("title_original")
            or news.get("title_en")
            or news.get("title")
            or ""
        ).strip()

        url = str(
            news.get("url") or ""
        ).strip()

        if not title or not url:
            continue

        key = ("news", url)

        if key in seen:
            continue

        seen.add(key)

        similarity = news.get("similarity")

        score = (
            round(float(similarity), 4)
            if isinstance(similarity, (int, float))
            else None
        )

        sources.append(
            {
                "type": "news",
                "title": title,
                "url": url,
                "score": score,
            }
        )

    return sources


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
            "sources": [],
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

        "sources": _build_sources(
            final_retrieval
        ),

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
