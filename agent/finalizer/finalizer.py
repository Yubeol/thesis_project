import json
import os
import re

from openai import OpenAI

from agent.prompts.finalizer import FINALIZER_SYSTEM_PROMPT
from agent.schemas import GapAnalysis
from agent.grounding import unsupported_numeric_claims


def _format_evidence(
    evidence: list[str],
    *,
    kind: str,
    max_items: int,
    max_chars_per_item: int = 1800,
) -> str:
    if not evidence:
        return "(none)"

    formatted = []

    for index, item in enumerate(evidence[:max_items], start=1):
        text = str(item).strip()

        if len(text) > max_chars_per_item:
            text = text[:max_chars_per_item].rstrip() + "..."

        label = f"[{kind} {index}]"
        if not re.match(rf"^\[{kind}\s+{index}\]", text, re.I):
            text = f"{label}\n{text}"

        formatted.append(text)

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
    """
    Transformer 1차 초안과 검색 근거를 이용해
    최종 학술 초안을 생성한다.

    이 함수는 내용 품질과 grounding만 책임진다.
    최종 한국어 글자 수 4500~4600자 보정은
    output_limiter.enforce_korean_char_limit()가 담당한다.
    """

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

    paper_evidence_text = _format_evidence(
        paper_evidence,
        kind="PAPER",
        max_items=15,
    )

    news_evidence_text = _format_evidence(
        news_evidence,
        kind="NEWS",
        max_items=8,
    )

    user_prompt = f"""
OUTPUT LANGUAGE:
{output_language}

IMPORTANT LANGUAGE INSTRUCTION:
The entire final draft MUST be written in {output_language}.
The language of the original draft or retrieved evidence does NOT determine
the output language.
Translate and synthesize evidence into {output_language} when necessary.

If OUTPUT LANGUAGE is Korean:
- Write all prose in Korean.
- Use the section headings "서론", "본론", and "결론".
- Do not write the body in English.
- Keep proper nouns such as TikTok, K-pop, BLACKPINK, YouTube, Instagram,
  platform names, and author names in their natural form when appropriate.

If OUTPUT LANGUAGE is English:
- Use the section headings "Introduction", "Body", and "Conclusion".

TITLE:
{title}

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question}

ORIGINAL TRANSFORMER FIRST DRAFT:
{draft}

IMPORTANT ROLE OF THE TRANSFORMER DRAFT:
The Transformer output is only a first draft.
It may contain awkward wording, repetition, malformed words,
mistranslated names, unsupported claims, or weak section structure.
Do NOT preserve those errors merely because they appear in the draft.
The supplied evidence is the factual authority.

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
- market success,
- fandom growth,
- positive causal effects.

Commercial, causal, directional, or future-oriented outcomes may only be
stated when the supplied evidence explicitly and directly supports them.

Do not make predictions about future growth, importance, expansion,
or success unless the supplied evidence explicitly contains a supported
forecast.

When evidence supports association, participation, visibility,
circulation, exposure, or cultural diffusion but not causation,
use cautious academic wording instead of claiming a proven causal effect.

ACADEMIC PAPER EVIDENCE:
{paper_evidence_text}

NEWS EVIDENCE:
{news_evidence_text}

STRUCTURE REQUIREMENT:
If OUTPUT LANGUAGE is Korean:
- Produce exactly three sections: 서론, 본론, 결론.
- 서론: explain the research background, problem, purpose,
  and research question.
- 본론: make this the longest section.
  Synthesize and compare multiple evidence items.
  Explain mechanisms, patterns, relationships, limitations,
  and contrasting evidence when supported.
- In the Body, include a concrete case from a paper when one directly
  addresses the research question. State the observed event or participants'
  accounts, the study's finding, and the limit of that finding. Do not turn
  a related but different event into evidence of the requested outcome.
- If the supplied papers contain no directly relevant case, say so briefly
  instead of inventing one or filling the space with repeated generalities.
- 결론: directly answer the research question using only supported claims.
  Summarize the body and do not introduce new evidence.

If OUTPUT LANGUAGE is English:
- Produce exactly three sections: Introduction, Body, Conclusion.

QUALITY REQUIREMENT:
- Do not repeat the same claim merely to make the draft longer.
- Prefer synthesis and comparison across evidence over repetitive summary.
- Clearly distinguish what the evidence supports from what remains uncertain.
- Do not invent facts, numbers, names, citations, causal claims,
  commercial claims, or future predictions.
- Add [PAPER n] or [NEWS n] immediately after each concrete case or factual
  finding, using the exact label of the supporting evidence above. Do not
  cite an item merely because it was retrieved. Preserve these labels in
  the final draft so the reference list can identify actually used sources.

REMINDER:
Your final response MUST be written entirely in {output_language}.

Use only claims that can be supported by the evidence above.
Return only the complete academic draft.
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
            + "The previous generated draft introduced numeric values that do "
            + "not appear in the supplied evidence: "
            + ", ".join(unsupported)
            + ". Remove those unsupported values. "
            + "Do not replace them with new numbers or new unsupported claims. "
            + "Return the complete corrected draft.\n\n"
            + "PREVIOUS GENERATED DRAFT:\n"
            + content
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
