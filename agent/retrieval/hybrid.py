# agent/retrieval/hybrid.py

from __future__ import annotations

from typing import Any

from rag.graph.graph_rag.retriever import search_related_papers
from rag.vector.news_rag.retriever import search_news_vector
from rag.vector.paper_rag.retriever import (
    search_paper_chunks_by_ids,
    search_papers_vector,
)


def _extract_paper_id(item: dict[str, Any]) -> int | None:
    """
    검색 결과에서 paper_id를 안전하게 추출한다.
    Graph RAG 반환 구조 변경에도 어느 정도 대응하도록
    후보 키를 순서대로 확인한다.
    """
    for key in (
        "paper_id",
        "related_paper_id",
        "target_paper_id",
        "id",
    ):
        value = item.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def _deduplicate_items(
    items: list[dict[str, Any]],
    *,
    id_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """
    검색 결과 중복 제거.

    chunk_id/news_id/article_id 등이 존재하면 해당 ID를 우선 사용하고,
    없으면 paper_id + text 계열 값을 조합한다.
    """
    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    for item in items:
        unique_value = None

        for key in id_keys:
            value = item.get(key)
            if value is not None:
                unique_value = f"{key}:{value}"
                break

        if unique_value is None:
            unique_value = "|".join(
                str(item.get(key, ""))
                for key in (
                    "paper_id",
                    "title",
                    "chunk_text",
                    "content",
                    "text",
                )
            )

        if unique_value in seen:
            continue

        seen.add(unique_value)
        results.append(item)

    return results


def _rank_paper_passages(
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Balance source diversity without hiding highly relevant later chunks."""
    ranked = sorted(
        papers,
        key=lambda item: float(item.get("similarity") or 0.0),
        reverse=True,
    )
    seen_counts: dict[str, int] = {}
    scored: list[tuple[float, int, dict[str, Any]]] = []

    for index, item in enumerate(ranked):
        paper_id = item.get("paper_id")
        key = str(paper_id) if paper_id is not None else str(item.get("chunk_id"))
        repeated = seen_counts.get(key, 0)
        seen_counts[key] = repeated + 1
        score = float(item.get("similarity") or 0.0) - 0.015 * min(repeated, 4)
        scored.append((score, index, item))

    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [item for _, _, item in scored]


def retrieve_hybrid(
    *,
    paper_queries: list[str],
    news_queries: list[str],
    paper_top_k: int = 5,
    news_top_k: int = 5,
    graph_limit: int = 10,
    graph_chunks_per_paper: int = 2,
    use_graph: bool = True,
    strict_graph: bool = False,
) -> dict[str, Any]:
    """
    1차/2차 RAG에서 공통으로 사용하는 Hybrid Retrieval.

    흐름:
    1. Paper Vector Search
    2. Vector 결과의 paper_id를 seed로 Graph 확장
    3. Graph에서 찾은 논문의 관련 chunk 재검색
    4. News Vector Search
    5. 중복 제거

    strict_graph=False:
        Neo4j 장애가 발생해도 Vector RAG 결과로 계속 진행한다.

    strict_graph=True:
        Graph RAG 오류가 발생하면 예외를 그대로 발생시킨다.
    """

    paper_queries = [
        query.strip()
        for query in paper_queries
        if query and query.strip()
    ]

    news_queries = [
        query.strip()
        for query in news_queries
        if query and query.strip()
    ]

    vector_papers: list[dict[str, Any]] = []
    graph_papers: list[dict[str, Any]] = []
    news_results: list[dict[str, Any]] = []

    graph_errors: list[str] = []

    # ---------------------------------------------------------
    # 1. Paper Vector RAG
    # ---------------------------------------------------------

    for query in paper_queries:
        results = search_papers_vector(
            query_en=query,
            top_k=paper_top_k,
        )

        vector_papers.extend(results)

    vector_papers = _deduplicate_items(
        vector_papers,
        id_keys=("chunk_id",),
    )

    # ---------------------------------------------------------
    # 2. Graph RAG
    # Vector RAG에서 검색된 논문을 seed로 관계 논문 확장
    # ---------------------------------------------------------

    if use_graph and vector_papers:

        seed_paper_ids = []

        for item in vector_papers:
            paper_id = _extract_paper_id(item)

            if paper_id is not None:
                seed_paper_ids.append(paper_id)

        seed_paper_ids = list(dict.fromkeys(seed_paper_ids))

        if seed_paper_ids:
            try:
                related_papers = search_related_papers(
                    paper_ids=seed_paper_ids,
                    limit=graph_limit,
                )

                related_paper_ids = []

                for item in related_papers:
                    paper_id = _extract_paper_id(item)

                    if paper_id is not None:
                        related_paper_ids.append(paper_id)

                related_paper_ids = list(
                    dict.fromkeys(related_paper_ids)
                )

                # Seed 논문과 중복되는 Graph 결과 제외
                seed_id_set = set(seed_paper_ids)

                related_paper_ids = [
                    paper_id
                    for paper_id in related_paper_ids
                    if paper_id not in seed_id_set
                ]

                # Graph가 찾아낸 논문에서 실제 텍스트 chunk 검색
                if related_paper_ids:
                    for query in paper_queries:
                        chunks = search_paper_chunks_by_ids(
                            query_en=query,
                            paper_ids=related_paper_ids,
                            top_k_per_paper=graph_chunks_per_paper,
                        )

                        graph_papers.extend(chunks)

            except Exception as exc:
                if strict_graph:
                    raise

                graph_errors.append(
                    f"{type(exc).__name__}: {exc}"
                )

    graph_papers = _deduplicate_items(
        graph_papers,
        id_keys=("chunk_id",),
    )

    # ---------------------------------------------------------
    # 3. Vector + Graph Paper 결과 병합
    # ---------------------------------------------------------

    papers = _rank_paper_passages(
        _deduplicate_items(
            vector_papers + graph_papers,
            id_keys=("chunk_id",),
        )
    )

    # ---------------------------------------------------------
    # 4. News Vector RAG
    # ---------------------------------------------------------

    for query in news_queries:
        results = search_news_vector(
            query_en=query,
            top_k=news_top_k,
        )

        news_results.extend(results)

    news_results = _deduplicate_items(
        news_results,
        id_keys=(
            "chunk_id",
            "news_id",
            "article_id",
        ),
    )
    news_results.sort(
        key=lambda item: float(item.get("similarity") or 0.0),
        reverse=True,
    )

    return {
        "papers": papers,
        "news": news_results,

        "debug": {
            "paper_queries": paper_queries,
            "news_queries": news_queries,

            "vector_paper_count": len(vector_papers),
            "graph_paper_count": len(graph_papers),
            "paper_count": len(papers),
            "news_count": len(news_results),

            "graph_errors": graph_errors,
        },
    }
