import json
import os

from openai import OpenAI

from agent.prompts.gap_analyzer import GAP_ANALYZER_SYSTEM_PROMPT
from agent.schemas import GapAnalysis


def _format_evidence(
    evidence: list[str],
    max_items: int = 10,
    max_chars_per_item: int = 1500,
) -> str:
    if not evidence:
        return "(none)"

    results = []

    for index, item in enumerate(evidence[:max_items], start=1):
        text = str(item).strip()

        if len(text) > max_chars_per_item:
            text = text[:max_chars_per_item].rstrip() + "..."

        results.append(f"[{index}]\n{text}")

    return "\n\n".join(results)


def analyze_gaps(
    *,
    title: str,
    topic: str,
    research_question: str,
    draft: str,
    paper_evidence: list[str],
    news_evidence: list[str],
) -> GapAnalysis:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    model = os.getenv(
        "OPENAI_LLM_MODEL",
        "gpt-4o-mini",
    )

    user_prompt = f"""
TITLE:
{title}

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question}

DRAFT:
{draft}

PAPER EVIDENCE:
{_format_evidence(paper_evidence)}

NEWS EVIDENCE:
{_format_evidence(news_evidence)}

EVIDENCE COVERAGE:
- paper evidence items: {len(paper_evidence)}
- news evidence items: {len(news_evidence)}
- If the topic concerns a current real-world event or platform practice and news
  evidence is absent, evaluate that absence as a possible evidence gap.
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": GAP_ANALYZER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Gap analyzer returned an empty response."
        )

    result = GapAnalysis(**json.loads(content))

    if not result.needs_additional_retrieval:
        result.paper_queries = []
        result.news_queries = []

    result.paper_queries = result.paper_queries[:3]
    result.news_queries = result.news_queries[:3]

    return result
