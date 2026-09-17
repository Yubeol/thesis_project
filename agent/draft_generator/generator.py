from transformer.inference import generate_draft


def generate_transformer_draft(
    *,
    title: str,
    topic: str | None = None,
    research_question: str | None = None,
    paper_evidence: list[str] | None = None,
    news_evidence: list[str] | None = None,
    instruction: str | None = None,
) -> str:
    """
    Transformer inference 호출을 담당하는 단일 진입점.

    실제 모델 호출은 이 파일에서만 수행한다.
    """

    paper_evidence = paper_evidence or []
    news_evidence = news_evidence or []

    if (
        not paper_evidence
        and not news_evidence
    ):
        raise ValueError(
            "Transformer에 전달할 "
            "paper_evidence 또는 news_evidence가 필요합니다."
        )

    draft = generate_draft(
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
        instruction=instruction,
    )

    if not isinstance(
        draft,
        str,
    ):
        raise TypeError(
            "generate_draft() 반환값은 str이어야 합니다. "
            f"현재 타입: {type(draft).__name__}"
        )

    draft = draft.strip()

    if not draft:
        raise RuntimeError(
            "Transformer가 빈 초안을 반환했습니다."
        )

    return draft