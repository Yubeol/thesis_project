import re

from agent import run_agent_pipeline


def _extract_section(
    text: str,
    start_heading: str,
    end_heading: str | None = None,
) -> str:
    """
    Markdown 최종 논문에서 특정 섹션을 추출한다.
    """

    if end_heading:
        pattern = (
            rf"##\s*{re.escape(start_heading)}\s*"
            rf"(.*?)"
            rf"(?=##\s*{re.escape(end_heading)}|\Z)"
        )
    else:
        pattern = (
            rf"##\s*{re.escape(start_heading)}\s*"
            rf"(.*)"
        )

    match = re.search(
        pattern,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return ""

    return match.group(1).strip()


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

    # 제목
    title_match = re.search(
        r"^#\s+(.+)$",
        final_text,
        flags=re.MULTILINE,
    )

    if title_match:
        title = title_match.group(1).strip()
    else:
        title = fallback_title.strip()

    # 한국어 heading 우선
    introduction = _extract_section(
        final_text,
        "서론",
        "본론",
    )

    body = _extract_section(
        final_text,
        "본론",
        "결론",
    )

    conclusion = _extract_section(
        final_text,
        "결론",
    )

    # 혹시 Finalizer가 영어 heading을 반환한 경우도 대응
    if not introduction:
        introduction = _extract_section(
            final_text,
            "Introduction",
            "Body",
        )

    if not body:
        body = _extract_section(
            final_text,
            "Body",
            "Conclusion",
        )

    if not conclusion:
        conclusion = _extract_section(
            final_text,
            "Conclusion",
        )

    draft = {
        "title": title,
        "introduction": introduction,
        "body": body,
        "conclusion": conclusion,
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
        fallback_title=title_ko,
    )

    return {
        "status": "completed",
        "draft": draft,
        "sources": result.get(
            "sources",
            [],
        ),
        "message": None,
    }