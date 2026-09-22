import re
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
from agent.finalizer.case_validator import detect_case_contradictions
from agent.visualizer import extract_visuals
from agent.finalizer.output_limiter import (
    enforce_korean_char_limit,
)


def _contains_hangul(
    text: str,
) -> bool:
    """
    문자열에 한글이 포함되어 있는지 확인한다.
    """

    return any(
        "\uac00" <= char <= "\ud7a3"
        for char in (text or "")
    )


def _split_korean_final_draft(
    text: str,
) -> dict[str, str]:
    """
    Finalizer가 반환한 한국어 문자열을
    서론 / 본론 / 결론으로 분리한다.

    다음 형식을 모두 지원한다.

    서론
    ...

    # 서론
    ...

    ## 서론:
    ...
    """

    cleaned = (
        text
        or ""
    ).strip()

    if not cleaned:
        raise RuntimeError(
            "Finalizer 결과가 비어 있습니다."
        )

    cleaned = re.sub(
        r"^```(?:markdown|text)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    section_pattern = re.compile(
        r"(?im)"
        r"^[ \t]*"
        r"#{0,3}"
        r"[ \t]*"
        r"(서론|본론|결론)"
        r"[ \t]*"
        r"[:：]?"
        r"[ \t]*$"
    )

    matches = list(
        section_pattern.finditer(
            cleaned
        )
    )

    if len(matches) < 3:
        raise RuntimeError(
            "Finalizer 결과에서 "
            "서론/본론/결론을 찾을 수 없습니다."
        )

    sections: dict[str, str] = {}

    for index, match in enumerate(
        matches
    ):
        section_name = (
            match.group(1)
        )

        if section_name in sections:
            continue

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(cleaned)
        )

        section_text = (
            cleaned[
                match.end():end
            ]
            .strip()
        )

        sections[
            section_name
        ] = section_text

    required_sections = (
        "서론",
        "본론",
        "결론",
    )

    if not all(
        sections.get(section)
        for section
        in required_sections
    ):
        raise RuntimeError(
            "Finalizer 결과의 "
            "서론/본론/결론 중 "
            "비어 있는 부분이 있습니다."
        )

    return {
        "introduction":
            sections["서론"],

        "body":
            sections["본론"],

        "conclusion":
            sections["결론"],
    }


def _combine_korean_sections(
    draft_ko: dict[str, str],
) -> str:
    """
    Backend가 기존처럼 사용할 수 있도록
    서론/본론/결론 문자열로 다시 합친다.

    제목은 별도 필드로 전달되므로
    final 문자열에는 중복해서 넣지 않는다.
    """

    return (
        "서론\n"
        f"{draft_ko['introduction'].strip()}"
        "\n\n"
        "본론\n"
        f"{draft_ko['body'].strip()}"
        "\n\n"
        "결론\n"
        f"{draft_ko['conclusion'].strip()}"
    )


def _limit_korean_final_draft(
    *,
    title: str,
    topic: str,
    research_question: str,
    final_text: str,
) -> str:
    """
    한국어 최종 논문이 최소 4500자 이상이 되도록 보정한다.

    최대 글자 수는 제한하지 않는다.

    output_limiter는 내부적으로 title까지 포함해
    글자 수를 세기 때문에 여기서는 title을 빈 문자열로
    전달해 논문 본문 자체의 길이를 맞춘다.
    """

    language_source = (
        f"{title} "
        f"{topic} "
        f"{research_question}"
    )

    if not _contains_hangul(
        language_source
    ):
        return final_text.strip()

    draft_sections = (
        _split_korean_final_draft(
            final_text
        )
    )

    limiter_input = {
        "title": "",

        "introduction":
            draft_sections[
                "introduction"
            ],

        "body":
            draft_sections[
                "body"
            ],

        "conclusion":
            draft_sections[
                "conclusion"
            ],
    }

    limited = (
        enforce_korean_char_limit(
            limiter_input,
            min_chars=4502,
        )
    )

    final = (
        _combine_korean_sections(
            limited
        )
    )

    final_length = len(
        final
    )

    if final_length < 4500:
        raise RuntimeError(
            "최종 한국어 초안이 "
            "4500자 이상이 아닙니다: "
            f"{final_length}자"
        )

    return final


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

    topic = (
        topic_ko
        or title_ko
    )

    analysis = analyze_query(
        title=title_ko,
        topic=topic,
        research_question="",
        instruction="",
    )

    if not analysis.allowed:
        return {
            "allowed": False,
            "rejection_reason":
                analysis.rejection_reason,
            "analysis":
                analysis.model_dump(),
            "retrieval": None,
            "paper_evidence": [],
            "news_evidence": [],
        }

    retrieval = retrieve_hybrid(
        paper_queries=
            analysis.paper_queries,

        news_queries=
            analysis.news_queries,

        use_graph=use_graph,
        strict_graph=strict_graph,
    )

    paper_evidence, news_evidence = (
        build_evidence_lists(
            retrieval
        )
    )

    return {
        "allowed": True,
        "rejection_reason": None,
        "analysis":
            analysis.model_dump(),
        "retrieval":
            retrieval,
        "paper_evidence":
            paper_evidence,
        "news_evidence":
            news_evidence,
    }


def _build_sources(
    retrieval: dict[str, Any],
    *,
    cited_text: str = "",
) -> list[dict[str, str]]:
    """
    최종 RAG 검색 결과에서 Frontend에 노출할 근거 목록을 생성한다.
    최종 글에 [PAPER n]/[NEWS n] 인용이 있으면 실제 인용된 항목만
    노출한다. 구형 출력에 인용 표지가 없으면 기존 목록 동작을 유지한다.
    """

    sources: list[
        dict[str, str]
    ] = []

    seen: set[
        tuple[str, str]
    ] = set()

    citations = {
        (kind.upper(), int(index))
        for kind, index in re.findall(
            r"\[\s*(PAPER|NEWS)\s+(\d+)\s*\]",
            cited_text,
            flags=re.I,
        )
    }

    papers = retrieval.get("papers", [])
    news_items = retrieval.get("news", [])

    if citations:
        # Evidence labels are assigned only to passages with body text.
        papers = [
            item
            for index, item in enumerate(
                (paper for paper in papers if str(paper.get("content") or "").strip()),
                start=1,
            )
            if ("PAPER", index) in citations
        ]
        news_items = [
            item
            for index, item in enumerate(
                (news for news in news_items if str(news.get("content") or "").strip()),
                start=1,
            )
            if ("NEWS", index) in citations
        ]

    # -------------------------
    # Papers
    # -------------------------

    for paper in papers:
        title = str(
            paper.get("title") or ""
        ).strip()

        url = str(
            paper.get(
                "source_url"
            )
            or ""
        ).strip()

        if not url:
            doi = str(
                paper.get("doi") or ""
            ).strip()

            if doi:
                doi = (
                    doi
                    .removeprefix(
                        "doi:"
                    )
                    .strip()
                )

                if doi.startswith(
                    (
                        "http://",
                        "https://",
                    )
                ):
                    url = doi
                else:
                    url = (
                        f"https://doi.org/{doi}"
                    )

        if not title or not url:
            continue

        key = (
            "paper",
            url,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "type":
                    "paper",

                "title":
                    title,

                "url":
                    url,
            }
        )

    # -------------------------
    # News
    # -------------------------

    for news in news_items:
        title = str(
            news.get(
                "title_original"
            )
            or news.get(
                "title_en"
            )
            or news.get(
                "title"
            )
            or ""
        ).strip()

        url = str(
            news.get("url") or ""
        ).strip()

        if not title or not url:
            continue

        key = (
            "news",
            url,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "type":
                    "news",

                "title":
                    title,

                "url":
                    url,
            }
        )

    return sources


def _align_citations_to_sources(
    final_text: str,
    retrieval: dict[str, Any],
    sources: list[dict[str, str]],
) -> str:
    """Match in-text evidence labels to the displayed reference numbers."""
    source_numbers = {
        (source["type"], source["url"]): index
        for index, source in enumerate(sources, start=1)
    }
    labels: dict[tuple[str, int], int] = {}

    paper_items = [
        item for item in retrieval.get("papers", [])
        if str(item.get("content") or "").strip()
    ]
    for index, item in enumerate(paper_items, start=1):
        url = str(item.get("source_url") or "").strip()
        if not url:
            doi = str(item.get("doi") or "").removeprefix("doi:").strip()
            if doi:
                url = doi if doi.startswith(("http://", "https://")) else f"https://doi.org/{doi}"
        number = source_numbers.get(("paper", url))
        if number:
            labels[("PAPER", index)] = number

    news_items = [
        item for item in retrieval.get("news", [])
        if str(item.get("content") or "").strip()
    ]
    for index, item in enumerate(news_items, start=1):
        number = source_numbers.get(("news", str(item.get("url") or "").strip()))
        if number:
            labels[("NEWS", index)] = number

    def replace(match: re.Match[str]) -> str:
        key = (match.group(1).upper(), int(match.group(2)))
        number = labels.get(key)
        return f"[{number}]" if number else match.group(0)

    return re.sub(r"\[\s*(PAPER|NEWS)\s+(\d+)\s*\]", replace, final_text, flags=re.I)

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
    → 한국어 최소 4500자 보정
    """

    analysis = analyze_query(
        title=title,
        topic=topic,
        research_question=
            research_question,
        instruction=instruction,
    )

    if not analysis.allowed:
        return {
            "allowed": False,
            "sources": [],
            "visuals": [],
            "rejection_reason":
                analysis.rejection_reason,

            "title":
                analysis.title,

            "topic":
                analysis.topic,

            "research_question":
                analysis.research_question,

            "draft": None,
            "final": None,
            "gap_analysis": None,

            "adaptive_retrieval_performed":
                False,

            "evidence_count": {
                "initial_papers": 0,
                "initial_news": 0,
                "final_papers": 0,
                "final_news": 0,
            },

            "retrieval_debug": {},
        }

    initial_retrieval = (
        retrieve_hybrid(
            paper_queries=
                analysis.paper_queries,

            news_queries=
                analysis.news_queries,

            use_graph=use_graph,

            strict_graph=
                strict_graph,
        )
    )

    paper_evidence, news_evidence = (
        build_evidence_lists(
            initial_retrieval
        )
    )

    transformer_paper_evidence, \
        transformer_news_evidence = (
            build_evidence_lists(
                initial_retrieval,

                max_chars_per_item=
                    1200,

                max_papers=4,
                max_news=2,
            )
        )

    draft = (
        generate_transformer_draft(
            title=
                analysis.title,

            topic=
                analysis.topic,

            research_question=
                analysis.research_question,

            paper_evidence=
                transformer_paper_evidence,

            news_evidence=
                transformer_news_evidence,

            instruction=
                analysis.instruction,
        )
    )

    gap_analysis = analyze_gaps(
        title=
            analysis.title,

        topic=
            analysis.topic,

        research_question=
            analysis.research_question,

        draft=draft,

        paper_evidence=
            paper_evidence,

        news_evidence=
            news_evidence,
    )

    adaptive_result = (
        run_adaptive_retrieval(
            gap_analysis=
                gap_analysis,

            current_retrieval=
                initial_retrieval,

            use_graph=
                use_graph,

            strict_graph=
                strict_graph,
        )
    )

    final_retrieval = (
        adaptive_result[
            "retrieval"
        ]
    )

    final_paper_evidence, \
        final_news_evidence = (
            build_evidence_lists(
                final_retrieval
            )
        )

    final = finalize_draft(
        title=
            analysis.title,

        topic=
            analysis.topic,

        research_question=
            analysis.research_question,

        draft=
            draft,

        gap_analysis=
            gap_analysis,

        paper_evidence=
            final_paper_evidence,

        news_evidence=
            final_news_evidence,
    )

    final = (
        _limit_korean_final_draft(
            title=
                analysis.title,

            topic=
                analysis.topic,

            research_question=
                analysis.research_question,

            final_text=
                final,
        )
    )

    contradictions = detect_case_contradictions(
        title=analysis.title,
        draft=final,
        paper_evidence=final_paper_evidence,
    )
    if contradictions:
        final = finalize_draft(
            title=analysis.title,
            topic=analysis.topic,
            research_question=analysis.research_question,
            draft=draft,
            gap_analysis=gap_analysis,
            paper_evidence=final_paper_evidence,
            news_evidence=final_news_evidence,
            correction_notes=contradictions,
        )
        final = _limit_korean_final_draft(
            title=analysis.title,
            topic=analysis.topic,
            research_question=analysis.research_question,
            final_text=final,
        )
        remaining = detect_case_contradictions(
            title=analysis.title,
            draft=final,
            paper_evidence=final_paper_evidence,
        )
        if remaining:
            raise RuntimeError(
                "Finalizer repeatedly reversed a cited paper's case outcome: "
                + "; ".join(remaining)
            )

    sources = _build_sources(
        final_retrieval,
        cited_text=final,
    )

    final = _align_citations_to_sources(final, final_retrieval, sources)

    visuals = extract_visuals(
        title=
        analysis.title,

        topic=
        analysis.topic,

        research_question=
        analysis.research_question,

        retrieval=
        final_retrieval,

        sources=
        sources,
    )

    return {
        "allowed": True,
        "rejection_reason": None,

        "title":
            analysis.title,

        "topic":
            analysis.topic,

        "research_question":
            analysis.research_question,

        "draft":
            draft,

        "final":
            final,

        "sources":
            sources,

        "visuals":
            visuals,

        "gap_analysis":
            gap_analysis.model_dump(),

        "adaptive_retrieval_performed":
            adaptive_result[
                "performed"
            ],

        "evidence_count": {
            "initial_papers":
                len(
                    initial_retrieval.get(
                        "papers",
                        [],
                    )
                ),

            "initial_news":
                len(
                    initial_retrieval.get(
                        "news",
                        [],
                    )
                ),

            "final_papers":
                len(
                    final_retrieval.get(
                        "papers",
                        [],
                    )
                ),

            "final_news":
                len(
                    final_retrieval.get(
                        "news",
                        [],
                    )
                ),
        },

        "retrieval_debug":
            final_retrieval.get(
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
        topic=(
            topic_ko
            or title_ko
        ),
    )

    return {
        **result,

        "status": (
            "completed"
            if result["allowed"]
            else "abstained"
        ),

        "transformer_draft":
            result.get(
                "draft"
            ),

        "final_text":
            result.get(
                "final"
            ),

        "message":
            result.get(
                "rejection_reason"
            ),
    }
