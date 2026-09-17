from transformer.inference import generate_draft


def generate_transformer_draft(
    *,
    title: str,
    topic: str,
    research_question: str,
    paper_evidence: list[str],
    news_evidence: list[str],
    instruction: str = "",
) -> str:
    draft = generate_draft(
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
        instruction=instruction,
    )

    if not isinstance(draft, str):
        raise TypeError(
            f"generate_draft() 반환값은 str이어야 합니다. "
            f"현재 타입: {type(draft).__name__}"
        )

    draft = draft.strip()

    if not draft:
        raise RuntimeError("Transformer가 빈 초안을 반환했습니다.")

    return draft