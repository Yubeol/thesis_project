import json

from agent.translator import (
    get_chat_model,
    get_openai_client,
)
from agent.transformer_adapter import (
    build_evidence_text,
)


def _parse_json_response(text: str) -> dict:
    """
    LLM이 반환한 JSON 문자열을 dict로 변환한다.
    혹시 ```json 코드블록이 포함되어도 제거한다.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace(
            "```json",
            "",
            1,
        )
        cleaned = cleaned.replace(
            "```",
            "",
        )
        cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LLM 최종 초안을 JSON으로 해석할 수 없습니다.\n"
            f"응답 내용:\n{cleaned}"
        ) from exc

    required_keys = {
        "introduction",
        "body",
        "conclusion",
    }

    missing = required_keys - result.keys()

    if missing:
        raise RuntimeError(
            "LLM 결과에 필수 항목이 없습니다: "
            + ", ".join(sorted(missing))
        )

    return result


def finalize_english_draft(
    title_en: str,
    topic_en: str | None,
    transformer_draft: str,
    retrieval: dict,
) -> dict:
    """
    Transformer 초안 + Hybrid RAG Evidence를 바탕으로
    Introduction / Body / Conclusion 구조의
    최종 영문 논문 초안을 생성한다.

    RAG Evidence를 최우선 근거로 사용하며,
    Evidence에 없는 사실을 새롭게 만들지 않는다.
    """

    if not title_en or not title_en.strip():
        raise ValueError(
            "영문 논문 제목이 비어 있습니다."
        )

    if not transformer_draft:
        raise ValueError(
            "Transformer 초안이 비어 있습니다."
        )

    evidence = build_evidence_text(
        retrieval
    )

    if not evidence:
        raise RuntimeError(
            "LLM에 전달할 Evidence가 없습니다."
        )

    client = get_openai_client()

    topic_text = (
        topic_en.strip()
        if topic_en
        else "Not separately provided."
    )

    prompt = f"""
Research Title:
{title_en}

Research Topic:
{topic_text}

Transformer Draft:
{transformer_draft}

Retrieved Evidence:
{evidence}

Using ONLY the retrieved evidence as the factual basis,
revise and organize the Transformer draft into an
academic paper draft.

Requirements:

1. Produce exactly three sections:
   - introduction
   - body
   - conclusion

2. The retrieved evidence is authoritative.
   If the Transformer draft contains claims that are not
   supported by the evidence, remove those claims.

3. Do not invent:
   - statistics
   - survey results
   - experiments
   - citations
   - authors
   - dates
   - references
   - findings

4. Do not claim that this project conducted original
   experiments or surveys.

5. Synthesize the retrieved evidence rather than merely
   listing it.

6. Maintain an academic and analytical tone.

7. Do not include Markdown headings.

8. Return ONLY valid JSON using exactly this format:

{{
  "introduction": "...",
  "body": "...",
  "conclusion": "..."
}}
""".strip()

    response = client.responses.create(
        model=get_chat_model(),
        instructions=(
            "You are an academic writing assistant. "
            "Use retrieved evidence conservatively and "
            "never fabricate unsupported information."
        ),
        input=prompt,
    )

    output = response.output_text.strip()

    if not output:
        raise RuntimeError(
            "LLM 최종 초안 결과가 비어 있습니다."
        )

    result = _parse_json_response(
        output
    )

    return {
        "introduction": (
            result["introduction"].strip()
        ),
        "body": (
            result["body"].strip()
        ),
        "conclusion": (
            result["conclusion"].strip()
        ),
    }