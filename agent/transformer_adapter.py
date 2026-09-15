import os

from dotenv import load_dotenv

load_dotenv()


def build_evidence_text(
    retrieval: dict,
) -> str:
    """
    Hybrid RAG 결과를 Transformer가 받을
    하나의 Evidence 텍스트로 변환한다.
    """

    sections = []

    # -----------------------------------------
    # Papers
    # -----------------------------------------

    papers = retrieval.get("papers", [])

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        chunks = paper.get(
            "evidence_chunks",
            [],
        )

        if not chunks:
            continue

        sections.append(
            f"[Paper {index}]"
        )

        sections.append(
            f"Title: {paper.get('title', '')}"
        )

        if paper.get("authors"):
            sections.append(
                f"Authors: {paper['authors']}"
            )

        if paper.get("published_year"):
            sections.append(
                f"Year: {paper['published_year']}"
            )

        for chunk in chunks:
            content = chunk.get("content")

            if content:
                section = chunk.get(
                    "section",
                    "unknown",
                )

                sections.append(
                    f"[{section}] {content}"
                )

        sections.append("")

    # -----------------------------------------
    # News
    # -----------------------------------------

    news_results = retrieval.get(
        "news",
        [],
    )

    for index, news in enumerate(
        news_results,
        start=1,
    ):
        content = news.get("content")

        if not content:
            continue

        sections.append(
            f"[News {index}]"
        )

        sections.append(
            f"Title: "
            f"{news.get('title_en', '')}"
        )

        sections.append(content)
        sections.append("")

    return "\n".join(sections).strip()


def _mock_generate_draft(
    title: str,
    topic: str | None,
    evidence: str,
) -> str:
    """
    Transformer 개발 완료 전
    Agent 연결 테스트용 Mock.
    """

    return (
        "[MOCK TRANSFORMER DRAFT]\n\n"
        f"Title: {title}\n"
        f"Topic: {topic or ''}\n\n"
        "Evidence-based draft placeholder.\n"
        f"Evidence length: {len(evidence)} characters."
    )


def generate_transformer_draft(
    title: str,
    topic: str | None,
    retrieval: dict,
) -> str:
    """
    Hybrid RAG Evidence를 Transformer에 전달해
    영문 초안을 생성한다.

    현재:
        Mock 사용 가능

    추후:
        transformer.inference.generate_draft()
        실제 함수 호출
    """

    evidence = build_evidence_text(
        retrieval
    )

    if not evidence:
        raise RuntimeError(
            "Transformer에 전달할 Evidence가 없습니다."
        )

    use_mock = (
        os.getenv(
            "TRANSFORMER_USE_MOCK",
            "true",
        ).lower()
        == "true"
    )

    # -----------------------------------------
    # 개발 중 Mock
    # -----------------------------------------

    if use_mock:
        return _mock_generate_draft(
            title=title,
            topic=topic,
            evidence=evidence,
        )

    # -----------------------------------------
    # 실제 Transformer
    # -----------------------------------------

    try:
        from transformer.inference import (
            generate_draft,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformer inference 모듈을 "
            "불러올 수 없습니다."
        ) from exc

    draft = generate_draft(
        title=title,
        topic=topic,
        evidence=evidence,
    )

    if not draft:
        raise RuntimeError(
            "Transformer 초안 결과가 비어 있습니다."
        )

    return draft