import json
import os

from openai import OpenAI

from agent.prompts.finalizer import FINALIZER_SYSTEM_PROMPT
from agent.schemas import GapAnalysis
from agent.grounding import unsupported_numeric_claims


def _format_evidence(
    evidence: list[str],
    *,
    max_items: int,
    max_chars_per_item: int = 1200,
) -> str:
    if not evidence:
        return "(none)"

    formatted = []

    for index, item in enumerate(evidence[:max_items], start=1):
        text = str(item).strip()

        if len(text) > max_chars_per_item:
            text = text[:max_chars_per_item].rstrip() + "..."

        formatted.append(
            f"[{index}]\n{text}"
        )

    return "\n\n".join(formatted)


def _detect_output_language(
    title: str,
    topic: str,
    research_question: str,
) -> str:
    source = f"{title} {topic} {research_question}"

    if any("\uac00" <= ch <= "\ud7a3" for ch in source):
        return "Korean"

    return "English"


def finalize_draft(
    *,
    title: str,
    topic: str,
    research_question: str,
    draft: str,
    gap_analysis: GapAnalysis,
    paper_evidence: list[str],
    news_evidence: list[str],
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    client = OpenAI(api_key=api_key)

    model = os.getenv(
        "OPENAI_LLM_MODEL",
        "gpt-4o-mini",
    )

    gaps = [
        gap.model_dump()
        for gap in gap_analysis.gaps
    ]

    restricted_claims = [
        {
            "claim": gap.claim,
            "reason": gap.reason,
            "evidence_type": gap.evidence_type,
        }
        for gap in gap_analysis.gaps
    ]

    output_language = _detect_output_language(
        title,
        topic,
        research_question,
    )

    user_prompt = f"""
OUTPUT LANGUAGE:
{output_language}

IMPORTANT LANGUAGE INSTRUCTION:
The entire final draft MUST be written in {output_language}.
The language of the original draft or retrieved evidence does NOT determine the output language.
Translate and synthesize evidence into {output_language} when necessary.

If OUTPUT LANGUAGE is Korean:
- Write all prose in Korean.
- Use the section headings "서론", "본론", and "결론".
- Do not write the body in English.
- Keep proper nouns such as TikTok, K-pop, BLACKPINK, and author names in their natural form when appropriate.

TITLE:
{title}

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question}

ORIGINAL DRAFT:
{draft}

IDENTIFIED EVIDENCE GAPS:
{json.dumps(
    gaps,
    ensure_ascii=False,
    indent=2,
)}

RESTRICTED CLAIMS:
{json.dumps(
    restricted_claims,
    ensure_ascii=False,
    indent=2,
)}

GAP RESOLUTION REQUIREMENT:
Every identified gap above MUST be explicitly resolved in the final draft.

If the retrieved evidence does not directly support a claim:
- weaken the claim,
- explicitly state that evidence is insufficient,
- or remove the claim.

Do NOT replace an unsupported claim with another unsupported claim.

STRICT GROUNDING RULE:
The claims listed under RESTRICTED CLAIMS were identified as unsupported
or weakly supported.

Do NOT:
- restate them as established facts,
- replace them with similar unsupported claims,
- broaden them into other causal claims,
- broaden them into commercial claims,
- make future predictions based on them.

Evidence of visibility, exposure, engagement, participation,
or cultural diffusion does NOT by itself prove:
- album sales,
- revenue growth,
- brand value,
- purchasing behavior,
- commercial opportunities,
- market success.

Commercial or causal outcomes may only be stated when the supplied
evidence explicitly and directly supports them.

Do not make predictions about future growth, importance, or success
unless the supplied evidence explicitly contains a supported forecast.

ACADEMIC PAPER EVIDENCE:
{_format_evidence(
    paper_evidence,
    max_items=15,
)}

NEWS EVIDENCE:
{_format_evidence(
    news_evidence,
    max_items=8,
)}

REMINDER:
Your final response MUST be written entirely in {output_language}.

Use only claims that can be supported by the evidence above.
Do not invent new factual, commercial, causal, or future-oriented claims.

Produce the final academic draft.
""".strip()

    def request(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": FINALIZER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Finalizer returned an empty response."
            )

        return content.strip()

    content = request(user_prompt)

    unsupported = unsupported_numeric_claims(
        content,
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
    )

    if unsupported:
        content = request(
            user_prompt
            + "\n\nCORRECTION REQUIRED:\n"
            + "The previous draft introduced numeric values that do not "
            + "appear in the supplied evidence: "
            + ", ".join(unsupported)
            + ". Remove those values and do not replace them with new "
            + "numbers. Return the complete corrected draft."
        )

        unsupported = unsupported_numeric_claims(
            content,
            title=title,
            topic=topic,
            research_question=research_question,
            paper_evidence=paper_evidence,
            news_evidence=news_evidence,
        )

    if unsupported:
        raise RuntimeError(
            "Finalizer repeatedly produced unsupported numeric claims: "
            + ", ".join(unsupported)
        )

    return content
