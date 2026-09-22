from __future__ import annotations

import re

from agent.translator import (
    get_chat_model,
    get_openai_client,
)


MIN_KOREAN_CHARS = 0
MAX_KOREAN_CHARS = 10000
TARGET_KOREAN_CHARS = 4800

MAX_REWRITE_ATTEMPTS = 4


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
    최종 사용자 출력 기준 글자 수를 계산한다.

    제목, 줄바꿈, '서론/본론/결론' heading,
    실제 본문을 모두 포함한다.
    """

    return len(
        combine_korean_draft(draft_ko)
    )


def _fixed_output_chars(
    title: str,
) -> int:
    """
    본문 세 부분이 모두 빈 문자열일 때도
    최종 출력에 항상 포함되는 고정 문자 수를 계산한다.

    즉 제목 + heading + 줄바꿈 문자 수다.
    """

    empty_draft = {
        "title": title,
        "introduction": "",
        "body": "",
        "conclusion": "",
    }

    return count_korean_chars(
        empty_draft
    )


def _split_rewritten_draft(
    text: str,
) -> dict:
    """
    LLM 보정 결과를 서론 / 본론 / 결론으로 분리한다.

    다음 형식을 모두 허용한다.

    서론
    ...

    # 서론
    ...

    ## 서론:
    ...
    """

    cleaned = (text or "").strip()

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

    pattern = re.compile(
        r"(?:^|\n)\s*#{0,3}\s*"
        r"(서론|본론|결론)\s*[:：]?\s*(?:\n|$)",
        flags=re.MULTILINE,
    )

    matches = list(
        pattern.finditer(cleaned)
    )

    sections = {}

    for index, match in enumerate(matches):
        name = match.group(1)

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(cleaned)
        )

        sections[name] = (
            cleaned[
                match.end():end
            ].strip()
        )

    required = (
        "서론",
        "본론",
        "결론",
    )

    if not all(
        sections.get(name)
        for name in required
    ):
        raise RuntimeError(
            "LLM 보정 결과를 "
            "서론/본론/결론으로 분리할 수 없습니다."
        )

    return {
        "introduction": sections["서론"],
        "body": sections["본론"],
        "conclusion": sections["결론"],
    }


def _rewrite_korean_draft_to_range(
    draft_ko: dict,
    *,
    min_chars: int,
    max_chars: int,
    target_chars: int,
) -> dict:
    """
    현재 초안의 사실과 논리 구조를 유지하면서
    전체 출력 글자 수를 지정 범위로 조정한다.

    너무 짧으면 확장하고,
    너무 길면 압축한다.
    """

    current_length = count_korean_chars(
        draft_ko
    )

    if (
        min_chars
        <= current_length
        <= max_chars
    ):
        return draft_ko

    title = draft_ko["title"].strip()

    fixed_chars = _fixed_output_chars(
        title
    )

    min_section_chars = max(
        1,
        min_chars - fixed_chars,
    )

    max_section_chars = max(
        min_section_chars,
        max_chars - fixed_chars,
    )

    target_section_chars = max(
        min_section_chars,
        min(
            target_chars - fixed_chars,
            max_section_chars,
        ),
    )

    current_section_chars = (
        len(draft_ko["introduction"].strip())
        + len(draft_ko["body"].strip())
        + len(draft_ko["conclusion"].strip())
    )

    client = get_openai_client()

    full_text = combine_korean_draft(
        draft_ko
    )

    if current_length < min_chars:
        difference = (
            target_chars - current_length
        )

        direction = "확장"

        length_instruction = (
            f"현재 최종 출력은 {current_length}자입니다. "
            f"목표는 약 {target_chars}자이며, "
            f"약 {max(0, difference)}자를 늘리는 방향으로 작성하세요."
        )

        editing_instruction = """
내용을 늘릴 때:
- 기존 근거의 의미를 더 자세히 설명
- 서로 다른 근거를 비교하고 연결
- 사례가 연구 질문과 어떤 관계가 있는지 분석
- 근거가 허용하는 범위에서 맥락과 한계를 설명
- 본론을 가장 많이 확장
- 이미 한 말을 다른 표현으로 반복하지 않기
""".strip()

    else:
        difference = (
            current_length - target_chars
        )

        direction = "압축"

        length_instruction = (
            f"현재 최종 출력은 {current_length}자입니다. "
            f"목표는 약 {target_chars}자이며, "
            f"약 {max(0, difference)}자를 줄이는 방향으로 작성하세요."
        )

        editing_instruction = """
내용을 줄일 때:
- 중복 주장부터 제거
- 같은 의미를 반복하는 문장을 통합
- 불필요한 수식어와 장황한 문장 축약
- 핵심 주장, 근거, 한계는 유지
- 문장 중간을 잘라내지 말고 자연스럽게 다시 작성
""".strip()

    response = client.responses.create(
        model=get_chat_model(),

        instructions=(
            "You are editing a Korean academic paper draft. "
            "Keep the draft strictly grounded in the supplied text. "
            "Do not add new facts, citations, statistics, people, "
            "organizations, studies, causal claims, commercial claims, "
            "or predictions."
        ),

        input=f"""
다음 한국어 학술 초안을 {direction}하여
최종 출력 글자 수를 정확한 범위 안으로 조정하세요.

현재 전체 글자 수:
{current_length}자

최종 전체 글자 수 조건:
- 최소 {min_chars}자
- 최대 {max_chars}자
- 목표 약 {target_chars}자

{length_instruction}

중요:
최종 시스템은 제목과 heading을 다시 붙입니다.
따라서 당신이 작성하는 세 본문의 글자 수 합계는
약 {target_section_chars}자를 목표로 하세요.

허용되는 세 본문의 범위는
약 {min_section_chars}자 이상
{max_section_chars}자 이하입니다.

현재 세 본문의 글자 수 합계는
{current_section_chars}자입니다.

출력 형식:
서론
[서론 본문]

본론
[본론 본문]

결론
[결론 본문]

반드시 지킬 조건:
- 제목은 출력하지 않음
- 서론 / 본론 / 결론 세 부분만 출력
- 본론이 가장 긴 부분이어야 함
- 기존 사실과 근거 유지
- 기존 초안에 없는 새로운 사실 추가 금지
- 새로운 숫자나 통계 추가 금지
- 새로운 인물, 기관, 논문, 기사 추가 금지
- 새로운 인과관계 추가 금지
- 새로운 상업적 효과 주장 추가 금지
- 근거 없는 미래 전망 추가 금지
- 같은 주장 반복 금지
- 같은 문장을 표현만 바꿔 반복하는 것도 금지
- 학술적 한국어 문체 유지
- 결과만 출력

길이 조정 방법:
{editing_instruction}

현재 논문 초안:
{full_text}
""".strip(),
    )

    rewritten = (
        response.output_text
        or ""
    ).strip()

    if not rewritten:
        raise RuntimeError(
            "한국어 초안 글자 수 보정 결과가 "
            "비어 있습니다."
        )

    result = _split_rewritten_draft(
        rewritten
    )

    # 제목은 LLM이 바꾸지 못하게 원본을 유지한다.
    result["title"] = draft_ko["title"]

    return result


def compress_korean_draft(
    draft_ko: dict,
    max_chars: int = MAX_KOREAN_CHARS,
) -> dict:
    """
    기존 코드와의 호환성을 위한 함수.

    max_chars를 초과할 때만 압축한다.
    기본 max_chars=4600일 때는 4500~4600자 범위를 목표로 한다.
    """

    current_length = count_korean_chars(
        draft_ko
    )

    if current_length <= max_chars:
        return draft_ko

    min_chars = min(
        MIN_KOREAN_CHARS,
        max_chars - 1,
    )

    min_chars = max(
        1,
        min_chars,
    )

    target_chars = (
        min_chars + max_chars
    ) // 2

    return _rewrite_korean_draft_to_range(
        draft_ko,
        min_chars=min_chars,
        max_chars=max_chars,
        target_chars=target_chars,
    )


def enforce_korean_char_limit(
    draft_ko: dict,
    min_chars: int = MIN_KOREAN_CHARS,
    max_chars: int = MAX_KOREAN_CHARS,
) -> dict:
    """
    최종 한국어 초안을 지정 범위로 보정한다.

    기본값:
    4500자 이상 ~ 4600자 이하

    이미 범위 안이면 LLM을 호출하지 않는다.
    범위를 벗어나면 최대 4회 보정한다.
    """

    if min_chars <= 0:
        raise ValueError(
            "min_chars must be positive."
        )

    if max_chars <= min_chars:
        raise ValueError(
            "max_chars must be greater than min_chars."
        )

    target_chars = (
        min_chars + max_chars
    ) // 2

    result = draft_ko

    for _ in range(
        MAX_REWRITE_ATTEMPTS
    ):
        current_length = count_korean_chars(
            result
        )

        if (
            min_chars
            <= current_length
            <= max_chars
        ):
            return result

        result = (
            _rewrite_korean_draft_to_range(
                result,
                min_chars=min_chars,
                max_chars=max_chars,
                target_chars=target_chars,
            )
        )

    final_length = count_korean_chars(
        result
    )

    if not (
        min_chars
        <= final_length
        <= max_chars
    ):
        raise RuntimeError(
            "최종 초안이 요구 글자 수 범위를 "
            "충족하지 못했습니다: "
            f"{final_length}자 "
            f"(요구 범위: "
            f"{min_chars}~{max_chars}자)"
        )

    return result