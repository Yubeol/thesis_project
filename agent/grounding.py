from __future__ import annotations

import re


NUMBER_PATTERN = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w])"
)


def numeric_claims(text: str) -> set[str]:
    return {
        match.group(0).replace(",", "")
        for match in NUMBER_PATTERN.finditer(text or "")
    }


def unsupported_numeric_claims(
    draft: str,
    *,
    title: str,
    topic: str,
    research_question: str,
    paper_evidence: list[str],
    news_evidence: list[str],
) -> list[str]:
    support = "\n".join(
        [
            title,
            topic,
            research_question,
            *paper_evidence,
            *news_evidence,
        ]
    )

    unsupported = numeric_claims(draft) - numeric_claims(support)
    return sorted(unsupported)
