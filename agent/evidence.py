# agent/evidence.py

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any


DEFAULT_MAX_CHARS_PER_ITEM = 5000

_FOCUS_STOPWORDS = {
    "about", "after", "against", "and", "are", "does", "from", "have",
    "how", "into", "kpop", "k-pop", "online", "that", "the", "their",
    "this", "what", "when", "with",
}
_CASE_SIGNALS = (
    "observ", "find", "found", "result", "compar", "differ", "protest",
    "refund", "collective", "collaborat", "support", "lack", "success",
    "interview", "case study", "반면", "사례", "관찰", "결과", "항의",
)
_COMPARISON_SIGNALS = (
    "instead", "lack", "success", "however", "on the other hand",
    "whereas", "more", "less", "stronger", "weaker", "반면", "달리",
)


def select_evidence_excerpt(
    item: str,
    *,
    max_chars: int,
    focus: str,
) -> str:
    """Keep query-relevant original sentences instead of only the passage head.

    Retrieved PDF chunks often place the observed result after background
    paragraphs. This selection never summarizes or adds facts; it retains the
    source header and extracts complete sentences from the supplied passage.
    """
    text = str(item).strip()
    if len(text) <= max_chars:
        return text

    marker = re.search(r"(?im)^Evidence:\s*\n", text)
    if marker is None:
        return text[:max_chars].rstrip() + "..."

    header = text[:marker.end()]
    body = text[marker.end():].strip()
    budget = max_chars - len(header) - 5
    if budget < 100:
        return text[:max_chars].rstrip() + "..."

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", body)
        if sentence.strip()
    ]
    if len(sentences) < 2:
        return header + body[:budget].rstrip() + "..."

    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", focus)
        if token.casefold() not in _FOCUS_STOPWORDS
    }
    selected: set[int] = set()
    covered: set[str] = set()
    remaining = budget

    while True:
        best_index = None
        best_score = 0
        for index, sentence in enumerate(sentences):
            if index in selected or len(sentence) + 3 > remaining:
                continue
            lowered = sentence.casefold()
            hits = {term for term in terms if term in lowered}
            signal_count = sum(signal in lowered for signal in _CASE_SIGNALS)
            score = (
                5 * len(hits - covered)
                + 2 * len(hits)
                + min(signal_count, 2)
                + (4 if hits and any(signal in lowered for signal in _COMPARISON_SIGNALS) else 0)
            )
            if score > best_score:
                best_index = index
                best_score = score

        if best_index is None or best_score == 0:
            break
        selected.add(best_index)
        covered.update(
            term for term in terms if term in sentences[best_index].casefold()
        )
        remaining -= len(sentences[best_index]) + 3

    if not selected:
        return header + body[:budget].rstrip() + "..."

    excerpt = " ... ".join(sentences[index] for index in sorted(selected))
    return header + excerpt


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    return " ".join(text.split())


def _format_authors(value: Any) -> str:
    """
    authors가 PostgreSQL에서 JSON 문자열 형태로 넘어오는 경우 처리.
    예:
    '["Jingran Liang", "Huizhong Miao"]'
    """
    if not value:
        return ""

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    if isinstance(value, str):
        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return ", ".join(str(item) for item in parsed)
        except json.JSONDecodeError:
            pass

        return value

    return str(value)


def _format_date(value: Any) -> str:
    if not value:
        return ""

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, date):
        return value.isoformat()

    return str(value)


def _truncate(
    text: str,
    max_chars: int,
) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def format_paper_evidence(
    papers: list[dict[str, Any]],
    *,
    max_chars_per_item: int = DEFAULT_MAX_CHARS_PER_ITEM,
    max_items: int | None = None,
) -> list[str]:
    """
    Paper Vector + Graph RAG 검색 결과를
    Transformer generate_draft()용 문자열 리스트로 변환한다.
    """

    results: list[str] = []

    # Hybrid retrieval already orders passages by relevance and paper diversity.
    # Number only usable passages so [PAPER n] matches the source index.
    for paper in papers:
        content = _clean_text(paper.get("content"))

        if not content:
            continue

        if max_items is not None and len(results) >= max_items:
            break

        index = len(results) + 1

        content = _truncate(
            content,
            max_chars=max_chars_per_item,
        )

        title = _clean_text(paper.get("title"))
        authors = _format_authors(paper.get("authors"))
        year = _clean_text(paper.get("published_year"))
        source = _clean_text(paper.get("source"))
        doi = _clean_text(paper.get("doi"))
        url = _clean_text(paper.get("source_url"))
        section = _clean_text(paper.get("section"))

        similarity = paper.get("similarity")

        lines = [
            f"[PAPER {index}]",
            f"Title: {title}",
        ]

        if authors:
            lines.append(f"Authors: {authors}")

        if year:
            lines.append(f"Published Year: {year}")

        if source:
            lines.append(f"Source: {source}")

        if doi:
            lines.append(f"DOI: {doi}")

        if url:
            lines.append(f"URL: {url}")

        if section:
            lines.append(f"Section: {section}")

        if isinstance(similarity, (int, float)):
            lines.append(f"Similarity: {similarity:.4f}")

        lines.extend(
            [
                "Evidence:",
                content,
            ]
        )

        results.append("\n".join(lines))

    return results


def format_news_evidence(
    news_items: list[dict[str, Any]],
    *,
    max_chars_per_item: int = DEFAULT_MAX_CHARS_PER_ITEM,
    max_items: int | None = None,
) -> list[str]:
    """
    News Vector RAG 검색 결과를
    Transformer generate_draft()용 문자열 리스트로 변환한다.
    """

    results: list[str] = []

    for news in news_items:
        content = _clean_text(news.get("content"))

        if not content:
            continue

        if max_items is not None and len(results) >= max_items:
            break

        index = len(results) + 1

        content = _truncate(
            content,
            max_chars=max_chars_per_item,
        )

        title = _clean_text(
            news.get("title_en")
            or news.get("title_original")
            or news.get("title")
        )

        source = _clean_text(news.get("source"))
        published_at = _format_date(news.get("published_at"))
        url = _clean_text(news.get("url"))

        similarity = news.get("similarity")

        lines = [
            f"[NEWS {index}]",
            f"Title: {title}",
        ]

        if source:
            lines.append(f"Source: {source}")

        if published_at:
            lines.append(f"Published At: {published_at}")

        if url:
            lines.append(f"URL: {url}")

        if isinstance(similarity, (int, float)):
            lines.append(f"Similarity: {similarity:.4f}")

        lines.extend(
            [
                "Evidence:",
                content,
            ]
        )

        results.append("\n".join(lines))

    return results


def build_evidence_lists(
    retrieval_result: dict[str, Any],
    *,
    max_chars_per_item: int = DEFAULT_MAX_CHARS_PER_ITEM,
    max_papers: int | None = None,
    max_news: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    retrieve_hybrid()의 반환값을 받아
    Transformer 입력 형식으로 한 번에 변환한다.

    returns:
        (
            paper_evidence,
            news_evidence,
        )
    """

    paper_evidence = format_paper_evidence(
        retrieval_result.get("papers", []),
        max_chars_per_item=max_chars_per_item,
        max_items=max_papers,
    )

    news_evidence = format_news_evidence(
        retrieval_result.get("news", []),
        max_chars_per_item=max_chars_per_item,
        max_items=max_news,
    )

    return paper_evidence, news_evidence
