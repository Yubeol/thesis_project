import os

from dotenv import load_dotenv

from agent.draft_generator.generator import (
    generate_transformer_draft as call_transformer,
)

load_dotenv()


def _clean(value) -> str:
    """
    RAG 결과의 문자열 값을 안전하게 정리한다.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple),
    ):
        return ", ".join(
            str(item).strip()
            for item in value
            if str(item).strip()
        )

    return str(
        value
    ).strip()


def build_paper_evidence(
    retrieval: dict,
) -> list[str]:
    """
    Hybrid RAG의 논문 검색 결과를
    Transformer paper_evidence 형식으로 변환한다.

    논문 한 편을 evidence item 하나로 구성한다.
    """

    results: list[str] = []

    papers = retrieval.get(
        "papers",
        [],
    )

    for paper in papers:
        chunks = paper.get(
            "evidence_chunks",
            [],
        )

        if not chunks:
            continue

        parts: list[str] = []

        title = _clean(
            paper.get(
                "title"
            )
        )

        authors = _clean(
            paper.get(
                "authors"
            )
        )

        published_year = _clean(
            paper.get(
                "published_year"
            )
        )

        if title:
            parts.append(
                f"Title: {title}"
            )

        if authors:
            parts.append(
                f"Authors: {authors}"
            )

        if published_year:
            parts.append(
                f"Year: {published_year}"
            )

        for chunk in chunks:
            content = _clean(
                chunk.get(
                    "content"
                )
            )

            if not content:
                continue

            section = _clean(
                chunk.get(
                    "section",
                    "unknown",
                )
            )

            if section:
                parts.append(
                    f"[{section}] {content}"
                )
            else:
                parts.append(
                    content
                )

        if parts:
            results.append(
                "\n".join(
                    parts
                )
            )

    return results


def build_news_evidence(
    retrieval: dict,
) -> list[str]:
    """
    Hybrid RAG의 뉴스 검색 결과를
    Transformer news_evidence 형식으로 변환한다.
    """

    results: list[str] = []

    news_results = retrieval.get(
        "news",
        [],
    )

    for news in news_results:
        content = _clean(
            news.get(
                "content"
            )
        )

        if not content:
            continue

        title = (
            _clean(
                news.get(
                    "title_en"
                )
            )
            or _clean(
                news.get(
                    "title"
                )
            )
        )

        published_at = (
            _clean(
                news.get(
                    "published_at"
                )
            )
            or _clean(
                news.get(
                    "published_date"
                )
            )
        )

        parts: list[str] = []

        if title:
            parts.append(
                f"Title: {title}"
            )

        if published_at:
            parts.append(
                f"Published: {published_at}"
            )

        parts.append(
            content
        )

        results.append(
            "\n".join(
                parts
            )
        )

    return results


def build_evidence_text(
    retrieval: dict,
) -> str:
    """
    기존 코드 및 디버깅 호환용.

    실제 Transformer 호출에서는 사용하지 않고
    paper/news evidence를 별도 리스트로 전달한다.
    """

    paper_evidence = (
        build_paper_evidence(
            retrieval
        )
    )

    news_evidence = (
        build_news_evidence(
            retrieval
        )
    )

    sections: list[str] = []

    for index, evidence in enumerate(
        paper_evidence,
        start=1,
    ):
        sections.append(
            f"[Paper {index}]"
        )
        sections.append(
            evidence
        )
        sections.append(
            ""
        )

    for index, evidence in enumerate(
        news_evidence,
        start=1,
    ):
        sections.append(
            f"[News {index}]"
        )
        sections.append(
            evidence
        )
        sections.append(
            ""
        )

    return "\n".join(
        sections
    ).strip()


def _mock_generate_draft(
    title: str,
    topic: str | None,
    paper_evidence: list[str],
    news_evidence: list[str],
) -> str:
    """
    Agent 연결 테스트용 Mock.
    """

    return (
        "[MOCK TRANSFORMER DRAFT]\n\n"
        f"Title: {title}\n"
        f"Topic: {topic or ''}\n\n"
        "Introduction:\n"
        "Mock introduction.\n\n"
        "Body:\n"
        "Mock body.\n\n"
        "Conclusion:\n"
        "Mock conclusion.\n\n"
        f"Paper evidence items: {len(paper_evidence)}\n"
        f"News evidence items: {len(news_evidence)}"
    )


def generate_transformer_draft(
    title: str,
    topic: str | None,
    retrieval: dict,
    research_question: str | None = None,
    instruction: str | None = None,
) -> str:
    """
    Hybrid RAG 결과를 Transformer 입력 형식으로 변환한다.

    실제 Transformer 모델 호출은
    agent/draft_generator/generator.py 에서만 수행한다.
    """

    paper_evidence = (
        build_paper_evidence(
            retrieval
        )
    )

    news_evidence = (
        build_news_evidence(
            retrieval
        )
    )

    if (
        not paper_evidence
        and not news_evidence
    ):
        raise RuntimeError(
            "Transformer에 전달할 "
            "Paper/News Evidence가 없습니다."
        )

    use_mock = (
        os.getenv(
            "TRANSFORMER_USE_MOCK",
            "false",
        )
        .strip()
        .lower()
        == "true"
    )

    if use_mock:
        return _mock_generate_draft(
            title=title,
            topic=topic,
            paper_evidence=paper_evidence,
            news_evidence=news_evidence,
        )

    return call_transformer(
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
        instruction=instruction,
    )