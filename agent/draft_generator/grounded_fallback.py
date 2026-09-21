from __future__ import annotations

import re


ABSTENTION = (
    "Insufficient evidence to generate this section reliably."
)

SECTION_NAMES = (
    "Introduction",
    "Body",
    "Conclusion",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _content_only(item: str) -> str:
    """Remove RAG metadata while retaining the retrieved passage."""
    marker = "Evidence:"

    if marker in item:
        return item.split(marker, 1)[1].strip()

    lines = []

    for line in item.splitlines():
        if re.match(
            r"^(?:\[|Title:|Authors?:|Published|Source:|DOI:|URL:|Similarity:)",
            line.strip(),
            flags=re.I,
        ):
            continue

        lines.append(line.strip())

    return " ".join(part for part in lines if part)


def _sentences(items: list[str]) -> list[str]:
    results = []
    seen = set()

    for item in items:
        content = " ".join(_content_only(item).split())
        content = re.sub(r"\b\d+(?=[A-Za-z])", "", content)
        content = re.sub(
            r"\b([A-Za-z]{2,})\s+-\s+([A-Za-z]{2,})\b",
            r"\1\2",
            content,
        )

        for sentence in re.split(r"(?<=[.!?])\s+", content):
            sentence = sentence.strip()

            if len(sentence.split()) < 7:
                continue

            if len(sentence.split()) > 60:
                continue

            if sentence.endswith("..."):
                continue

            if any(len(word) > 30 for word in sentence.split()):
                continue

            if not re.match(r"^[A-Z\"'([]", sentence):
                continue

            key = sentence.casefold()

            if key in seen:
                continue

            seen.add(key)
            results.append(sentence)

    return results


def _ranked_sentences(
    *,
    title: str,
    topic: str | None,
    research_question: str | None,
    paper_evidence: list[str],
    news_evidence: list[str],
) -> list[str]:
    query_terms = {
        token.casefold()
        for token in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9'-]+",
            " ".join(
                value
                for value in (title, topic, research_question)
                if value
            ),
        )
        if token.casefold() not in STOPWORDS
    }

    paper_sentences = _sentences(paper_evidence)
    news_sentences = _sentences(news_evidence)
    combined = [
        (sentence, 1)
        for sentence in paper_sentences
    ] + [
        (sentence, 0)
        for sentence in news_sentences
    ]

    ranked = sorted(
        enumerate(combined),
        key=lambda pair: (
            -len(
                query_terms
                & {
                    token.casefold()
                    for token in re.findall(
                        r"[A-Za-z0-9][A-Za-z0-9'-]+",
                        pair[1][0],
                    )
                }
            ),
            -pair[1][1],
            pair[0],
        ),
    )

    return [item[1][0] for item in ranked]


def build_grounded_fallback(
    *,
    title: str,
    topic: str | None,
    research_question: str | None,
    paper_evidence: list[str],
    news_evidence: list[str],
) -> str:
    """
    Build a conservative extractive draft when the trained model abstains.

    Retrieved sentences are copied verbatim. The only generated prose is a
    limitation statement, so this path cannot invent names, dates, or numbers.
    """
    ranked = _ranked_sentences(
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
    )

    if not ranked:
        return "\n\n".join(
            f"{name}:\n{ABSTENTION}"
            for name in SECTION_NAMES
        )

    introduction = " ".join(ranked[:2])
    body = " ".join(ranked[2:7] or ranked[:3])
    conclusion = (
        "The available evidence supports only the observations reported "
        "above; it does not by itself establish broader causal or "
        "commercial effects."
    )

    return (
        "Introduction:\n"
        + introduction
        + "\n\nBody:\n"
        + body
        + "\n\nConclusion:\n"
        + conclusion
    )


def replace_abstained_draft(
    draft: str,
    *,
    title: str,
    topic: str | None,
    research_question: str | None,
    paper_evidence: list[str],
    news_evidence: list[str],
    force: bool = False,
) -> tuple[str, bool]:
    matches = list(
        re.finditer(
            r"(?im)^(Introduction|Body|Conclusion):\s*",
            draft,
        )
    )
    sections = []

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(draft)
        sections.append(draft[match.end():end].strip())

    normalized_sections = {
        re.sub(r"\W+", " ", section).casefold().strip()
        for section in sections
        if section
    }
    normalized_inputs = {
        re.sub(r"\W+", " ", value or "").casefold().strip()
        for value in (title, topic, research_question)
    }

    malformed = (
        force
        or ABSTENTION in draft
        or len(matches) != len(SECTION_NAMES)
        or len(sections) != len(SECTION_NAMES)
        or any(len(section.split()) < 7 for section in sections)
        or len(normalized_sections) < 2
        or bool(normalized_sections & normalized_inputs)
    )

    if not malformed:
        return draft, False

    return (
        build_grounded_fallback(
            title=title,
            topic=topic,
            research_question=research_question,
            paper_evidence=paper_evidence,
            news_evidence=news_evidence,
        ),
        True,
    )
