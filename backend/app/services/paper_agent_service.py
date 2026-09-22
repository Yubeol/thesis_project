import re

from agent import run_agent_pipeline


SECTION_NAMES = {
    "서론": "introduction",
    "introduction": "introduction",
    "본론": "body",
    "body": "body",
    "결론": "conclusion",
    "conclusion": "conclusion",
}

SECTION_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?"
    r"(서론|본론|결론|Introduction|Body|Conclusion)"
    r"(?:\*\*)?[ \t]*[:：]?[ \t]*$",
    flags=re.MULTILINE | re.IGNORECASE,
)


def _parse_final_draft(
    final_text: str,
    fallback_title: str,
) -> dict[str, str]:
    """
    Agent Finalizer의 Markdown 결과를
    API 응답용 구조로 변환한다.
    """

    if not final_text or not final_text.strip():
        raise RuntimeError(
            "Agent 최종 결과가 비어 있습니다."
        )

    final_text = final_text.strip()

    headings = list(SECTION_HEADING.finditer(final_text))
    title_match = re.search(r"^#\s+(.+)$", final_text, flags=re.MULTILINE)
    title = fallback_title.strip()
    if title_match and (not headings or title_match.start() < headings[0].start()):
        title = title_match.group(1).strip()

    sections: dict[str, str] = {}
    for index, heading in enumerate(headings):
        name = SECTION_NAMES[heading.group(1).casefold()]
        if name in sections:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(final_text)
        sections[name] = final_text[heading.end():end].strip()

    draft = {
        "title": title,
        "introduction": sections.get("introduction", ""),
        "body": sections.get("body", ""),
        "conclusion": sections.get("conclusion", ""),
    }

    missing_fields = [
        field
        for field, value in draft.items()
        if not value
    ]

    if missing_fields:
        raise RuntimeError(
            "최종 초안 파싱에 실패했습니다. "
            "누락 항목: "
            + ", ".join(missing_fields)
        )

    return draft


def generate_paper(
    title_ko: str,
    topic_ko: str | None = None,
) -> dict:
    """
    Agent Pipeline을 호출하고
    Frontend API 계약 형태로 변환한다.
    """

    result = run_agent_pipeline(
        title_ko=title_ko,
        topic_ko=topic_ko,
    )

    status = result.get("status")

    # -----------------------------------------
    # 범위 밖 주제
    # -----------------------------------------

    if status == "abstained":
        return {
            "status": "abstained",
            "draft": None,
            "sources": [],
            "visuals": [],
            "message": (
                    result.get("message")
                    or result.get("rejection_reason")
                    or "해당 주제는 현재 서비스에서 지원하지 않습니다."
            ),
        }

    # -----------------------------------------
    # 예상하지 못한 Agent 상태
    # -----------------------------------------

    if status != "completed":
        raise RuntimeError(
            "Agent가 예상하지 못한 상태를 "
            f"반환했습니다: {status}"
        )

    # -----------------------------------------
    # 새 Orchestrator 최종 결과
    # -----------------------------------------

    final_text = (
        result.get("final")
        or result.get("final_text")
    )

    if not final_text:
        raise RuntimeError(
            "Agent가 completed 상태를 반환했지만 "
            "최종 논문 결과가 없습니다."
        )

    draft = _parse_final_draft(
        final_text=final_text,
        fallback_title=result.get("title") or title_ko,
    )

    return {
        "status": "completed",
        "draft": draft,
        "sources": result.get(
            "sources",
            [],
        ),
        "visuals": result.get(
            "visuals",
            [],
        ),
        "message": None,
    }
