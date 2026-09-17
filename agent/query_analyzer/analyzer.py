import json
import os

from openai import OpenAI

from agent.prompts.query_analyzer import QUERY_ANALYZER_SYSTEM_PROMPT
from agent.schemas import QueryAnalysis


def analyze_query(
    *,
    title: str = "",
    topic: str = "",
    research_question: str = "",
    instruction: str = "",
) -> QueryAnalysis:

    title = (title or "").strip()
    topic = (topic or "").strip()
    research_question = (research_question or "").strip()
    instruction = (instruction or "").strip()

    # 완전히 빈 입력은 LLM 호출도 하지 않고 바로 종료
    if not any([title, topic, research_question]):
        return QueryAnalysis(
            allowed=False,
            rejection_reason="논문 작성을 위한 제목, 주제 또는 연구 질문이 필요합니다.",
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=api_key)

    user_input = {
        "title": title,
        "topic": topic,
        "research_question": research_question,
        "instruction": instruction,
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": QUERY_ANALYZER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_input,
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("LLM 1이 빈 응답을 반환했습니다.")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM 1 응답을 JSON으로 변환하지 못했습니다: {content[:500]}"
        ) from exc

    analysis = QueryAnalysis.model_validate(result)

    # 사용자가 직접 입력한 값은 LLM이 임의 변경하지 못하도록 최종 보정
    if analysis.allowed:
        if title:
            analysis.title = title

        if topic:
            analysis.topic = topic

        if research_question:
            analysis.research_question = research_question

        if instruction:
            analysis.instruction = instruction

    return analysis