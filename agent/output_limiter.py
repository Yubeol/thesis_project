from agent.translator import (
    get_chat_model,
    get_openai_client,
)


MAX_KOREAN_CHARS = 4500


def combine_korean_draft(
    draft_ko: dict,
) -> str:
    """
    제목 + 서론 + 본론 + 결론을
    최종 출력 문자열로 합친다.
    """

    return (
        f"{draft_ko['title'].strip()}\n\n"
        "서론\n"
        f"{draft_ko['introduction'].strip()}\n\n"
        "본론\n"
        f"{draft_ko['body'].strip()}\n\n"
        "결론\n"
        f"{draft_ko['conclusion'].strip()}"
    )


def count_korean_chars(
    draft_ko: dict,
) -> int:
    """
    최종 출력 기준 글자 수를 계산한다.
    """
    return len(
        combine_korean_draft(draft_ko)
    )


def compress_korean_draft(
    draft_ko: dict,
    max_chars: int = MAX_KOREAN_CHARS,
) -> dict:
    """
    한국어 최종 초안이 글자 수 제한을 초과할 경우
    LLM을 이용해 의미와 근거를 유지하면서 압축한다.
    """

    current_length = count_korean_chars(
        draft_ko
    )

    if current_length <= max_chars:
        return draft_ko

    client = get_openai_client()

    full_text = combine_korean_draft(
        draft_ko
    )

    response = client.responses.create(
        model=get_chat_model(),
        instructions=(
            "You are editing a Korean academic paper draft. "
            "Shorten the text while preserving its factual "
            "claims, logical structure, and evidence. "
            "Do not add new facts, citations, statistics, "
            "or interpretations."
        ),
        input=f"""
다음 논문 초안을 전체 {max_chars}자 이하가 되도록 압축하세요.

조건:
- 서론 / 본론 / 결론 구조 유지
- 핵심 주장과 근거 유지
- 새로운 정보 추가 금지
- 중복 표현 우선 제거
- 학술적 문체 유지
- 결과만 출력

{full_text}
""".strip(),
    )

    compressed = response.output_text.strip()

    if not compressed:
        raise RuntimeError(
            "한국어 초안 압축 결과가 비어 있습니다."
        )


    result = _split_compressed_draft(
        compressed
    )

    result["title"] = draft_ko["title"]

    return result


def _split_compressed_draft(
    text: str,
) -> dict:
    """
    압축 결과를 서론 / 본론 / 결론으로 다시 분리한다.
    """

    try:
        intro_part, rest = text.split(
            "본론",
            1,
        )

        body_part, conclusion_part = (
            rest.split(
                "결론",
                1,
            )
        )

        introduction = intro_part.replace(
            "서론",
            "",
            1,
        ).strip()

        body = body_part.strip()
        conclusion = conclusion_part.strip()

    except ValueError as exc:
        raise RuntimeError(
            "압축 결과를 서론/본론/결론으로 "
            "분리할 수 없습니다."
        ) from exc

    return {
        "introduction": introduction,
        "body": body,
        "conclusion": conclusion,
    }


def enforce_korean_char_limit(
    draft_ko: dict,
    max_chars: int = MAX_KOREAN_CHARS,
) -> dict:
    """
    최종 한국어 초안을 max_chars 이하로 보정한다.

    필요 시 최대 2회 재압축한다.
    """

    result = draft_ko

    for _ in range(2):
        if count_korean_chars(result) <= max_chars:
            return result

        result = compress_korean_draft(
            result,
            max_chars=max_chars,
        )

    final_length = count_korean_chars(
        result
    )

    if final_length > max_chars:
        raise RuntimeError(
            "최종 초안이 글자 수 제한을 "
            f"충족하지 못했습니다: {final_length}자"
        )

    return result