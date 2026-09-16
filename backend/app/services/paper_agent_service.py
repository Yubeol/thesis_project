from agent import run_agent_pipeline


def generate_paper(
    title_ko: str,
    topic_ko: str | None = None,
) -> dict:
    """
    Agent Pipeline을 호출하고
    API가 사용하기 쉬운 형태로 변환한다.
    """

    result = run_agent_pipeline(
        title_ko=title_ko,
        topic_ko=topic_ko,
    )

    status = result.get("status")

    if status == "abstained":
        return {
            "status": "abstained",
            "draft": None,
            "character_count": 0,
            "message": result.get(
                "message",
                "충분한 근거를 찾지 못했습니다.",
            ),
        }

    if status != "completed":
        raise RuntimeError(
            "Agent가 예상하지 못한 상태를 "
            f"반환했습니다: {status}"
        )

    draft_ko = result.get("draft_ko")

    if not draft_ko:
        raise RuntimeError(
            "Agent가 completed 상태를 반환했지만 "
            "최종 한국어 초안이 없습니다."
        )

    required_fields = (
        "title",
        "introduction",
        "body",
        "conclusion",
    )

    missing_fields = [
        field
        for field in required_fields
        if not draft_ko.get(field)
    ]

    if missing_fields:
        raise RuntimeError(
            "최종 초안에 필요한 항목이 없습니다: "
            + ", ".join(missing_fields)
        )

    return {
        "status": "completed",
        "draft": {
            "title": draft_ko["title"],
            "introduction": draft_ko["introduction"],
            "body": draft_ko["body"],
            "conclusion": draft_ko["conclusion"],
        },
        "character_count": result.get(
            "character_count",
            0,
        ),
        "message": None,
    }