from typing import Any

from agent.retrieval import retrieve_hybrid
from agent.schemas import GapAnalysis


def _deduplicate_items(
    items: list[dict[str, Any]],
    *,
    item_type: str,
) -> list[dict[str, Any]]:
    seen = set()
    result = []

    for item in items:
        if item_type == "paper":
            key = (
                item.get("chunk_id")
                or (
                    item.get("paper_id"),
                    item.get("chunk_index"),
                )
                or item.get("content")
            )
        else:
            key = (
                item.get("chunk_id")
                or (
                    item.get("news_id"),
                    item.get("chunk_index"),
                )
                or item.get("content")
            )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def run_adaptive_retrieval(
    *,
    gap_analysis: GapAnalysis,
    current_retrieval: dict,
    use_graph: bool = True,
    strict_graph: bool = False,
) -> dict:
    """
    LLM2 결과에 따라 필요한 경우에만 2차 Hybrid RAG를 수행한다.

    반환값:
    {
        "performed": bool,
        "additional_retrieval": dict | None,
        "retrieval": 병합된 최종 retrieval 결과
    }
    """

    needs_retrieval = (
        gap_analysis.needs_additional_retrieval
        and (
            gap_analysis.paper_queries
            or gap_analysis.news_queries
        )
    )

    if not needs_retrieval:
        return {
            "performed": False,
            "additional_retrieval": None,
            "retrieval": current_retrieval,
        }

    additional = retrieve_hybrid(
        paper_queries=gap_analysis.paper_queries,
        news_queries=gap_analysis.news_queries,
        use_graph=use_graph,
        strict_graph=strict_graph,
    )

    merged_papers = _deduplicate_items(
        [
            *current_retrieval.get("papers", []),
            *additional.get("papers", []),
        ],
        item_type="paper",
    )

    merged_news = _deduplicate_items(
        [
            *current_retrieval.get("news", []),
            *additional.get("news", []),
        ],
        item_type="news",
    )
    merged_news.sort(
        key=lambda item: (
            float(
                item.get("ranking_score")
                or item.get("similarity")
                or 0.0
            ),
            float(item.get("similarity") or 0.0),
        ),
        reverse=True,
    )

    merged = {
        "papers": merged_papers,
        "news": merged_news,
        "debug": {
            "initial": current_retrieval.get(
                "debug",
                {},
            ),
            "additional": additional.get(
                "debug",
                {},
            ),
            "merged_paper_count": len(
                merged_papers
            ),
            "merged_news_count": len(
                merged_news
            ),
        },
    }

    return {
        "performed": True,
        "additional_retrieval": additional,
        "retrieval": merged,
    }
