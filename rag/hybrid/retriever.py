from rag.graph import search_related_papers
from rag.vector import (
    search_news_vector,
    search_paper_chunks_by_ids,
    search_papers_vector,
)

DEFAULT_RRF_K = 60


def _collapse_vector_papers(
    vector_results: list[dict],
) -> list[dict]:
    """
    Vector RAG는 Chunk 단위 결과를 반환한다.

    같은 논문의 여러 Chunk가 검색될 수 있으므로
    paper_id 기준으로 묶고 Chunk 정보는 evidence_chunks에 보관한다.
    """

    papers: dict[int, dict] = {}

    for rank, result in enumerate(vector_results, start=1):
        paper_id = result["paper_id"]

        if paper_id not in papers:
            papers[paper_id] = {
                "paper_id": paper_id,
                "title": result.get("title"),
                "authors": result.get("authors"),
                "published_year": result.get("published_year"),
                "source": result.get("source"),
                "source_url": result.get("source_url"),
                "doi": result.get("doi"),

                # 처음 등장한 순위가 가장 높은 Vector 순위
                "vector_rank": rank,

                "vector_similarity": result.get(
                    "similarity",
                    0.0,
                ),

                "evidence_chunks": [],
            }

        # 같은 논문의 검색된 Chunk들을 근거로 보존
        papers[paper_id]["evidence_chunks"].append(
            {
                "chunk_id": result.get("chunk_id"),
                "section": result.get("section"),
                "chunk_index": result.get("chunk_index"),
                "content": result.get("content"),
                "similarity": result.get("similarity"),
            }
        )

        current_similarity = (
            papers[paper_id]["vector_similarity"]
            or 0.0
        )

        new_similarity = result.get("similarity") or 0.0

        if new_similarity > current_similarity:
            papers[paper_id]["vector_similarity"] = (
                new_similarity
            )

    return list(papers.values())


def fuse_paper_results(
    vector_results: list[dict],
    graph_results: list[dict],
    top_k: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """
    Vector RAG와 Graph RAG 결과를 RRF 방식으로 통합한다.

    Vector similarity와 Graph relation_score는 서로 척도가 다르므로
    raw score를 직접 더하지 않고 검색 순위를 이용한다.
    """

    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    vector_papers = _collapse_vector_papers(
        vector_results
    )

    fused: dict[int, dict] = {}

    # -----------------------------------------------------
    # Vector RAG 결과
    # -----------------------------------------------------

    for rank, paper in enumerate(
        vector_papers,
        start=1,
    ):
        paper_id = paper["paper_id"]

        fused[paper_id] = {
            **paper,

            "graph_rank": None,
            "graph_relation_score": None,
            "relation_paths": [],

            "retrieval_sources": ["vector"],

            "rrf_score": (
                1.0 / (rrf_k + rank)
            ),
        }

    # -----------------------------------------------------
    # Graph RAG 결과
    # -----------------------------------------------------

    for rank, result in enumerate(
        graph_results,
        start=1,
    ):
        paper_id = result["paper_id"]

        graph_rrf = 1.0 / (rrf_k + rank)

        # Vector에서도 검색된 논문
        if paper_id in fused:
            fused[paper_id]["rrf_score"] += graph_rrf

            fused[paper_id]["graph_rank"] = rank

            fused[paper_id][
                "graph_relation_score"
            ] = result.get("relation_score")

            fused[paper_id][
                "relation_paths"
            ] = result.get(
                "relation_paths",
                [],
            )

            fused[paper_id][
                "retrieval_sources"
            ].append("graph")

        # Graph에서만 새롭게 발견된 논문
        else:
            fused[paper_id] = {
                "paper_id": paper_id,
                "title": result.get("title"),
                "authors": None,
                "published_year": result.get(
                    "published_year"
                ),
                "source": None,
                "source_url": None,
                "doi": None,

                "vector_rank": None,
                "vector_similarity": None,

                # Graph-only 논문은 이후 PostgreSQL에서
                # 본문 Chunk를 가져오면 된다.
                "evidence_chunks": [],

                "graph_rank": rank,
                "graph_relation_score": result.get(
                    "relation_score"
                ),
                "relation_paths": result.get(
                    "relation_paths",
                    [],
                ),

                "retrieval_sources": ["graph"],

                "rrf_score": graph_rrf,
            }

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # 최종 순위 추가
    for final_rank, item in enumerate(
        ranked_results,
        start=1,
    ):
        item["hybrid_rank"] = final_rank

    return ranked_results[:top_k]


def hybrid_retrieve(
    query_en: str,
    paper_vector_top_k: int = 10,
    news_top_k: int = 5,
    graph_top_k: int = 10,
    final_paper_top_k: int = 10,
) -> dict:
    """
    전체 Hybrid RAG 검색.

    1. 논문 Vector RAG
    2. 뉴스 Vector RAG
    3. Vector 결과의 paper_id로 Graph RAG 확장
    4. Vector + Graph 논문 결과 RRF 통합
    """

    if not query_en or not query_en.strip():
        raise ValueError(
            "Hybrid RAG Query가 비어 있습니다."
        )

    # -----------------------------------------------------
    # 1. Paper Vector RAG
    # -----------------------------------------------------

    paper_vector_results = search_papers_vector(
        query_en=query_en,
        top_k=paper_vector_top_k,
    )

    # -----------------------------------------------------
    # 2. News Vector RAG
    # -----------------------------------------------------

    news_results = search_news_vector(
        query_en=query_en,
        top_k=news_top_k,
    )

    # -----------------------------------------------------
    # 3. Graph RAG Seed 생성
    # -----------------------------------------------------

    seed_paper_ids = list(
        dict.fromkeys(
            result["paper_id"]
            for result in paper_vector_results
        )
    )

    graph_results = []

    if seed_paper_ids:
        graph_results = search_related_papers(
            paper_ids=seed_paper_ids,
            limit=graph_top_k,
        )

    # -----------------------------------------------------
    # 4. Hybrid Fusion
    # -----------------------------------------------------

    hybrid_papers = fuse_paper_results(
        vector_results=paper_vector_results,
        graph_results=graph_results,
        top_k=final_paper_top_k,
    )

    hybrid_papers = enrich_graph_only_papers(
        query_en=query_en,
        hybrid_papers=hybrid_papers,
        top_k_per_paper=2,
    )

    # 뉴스는 현재 Graph DB 대상이 아니므로
    # Vector 결과 그대로 반환
    return {
        "query_en": query_en,

        "papers": hybrid_papers,

        "news": news_results,

        "stats": {
            "paper_vector_count": len(
                paper_vector_results
            ),
            "graph_count": len(
                graph_results
            ),
            "hybrid_paper_count": len(
                hybrid_papers
            ),
            "news_count": len(
                news_results
            ),
        },
    }

def enrich_graph_only_papers(
    query_en: str,
    hybrid_papers: list[dict],
    top_k_per_paper: int = 2,
) -> list[dict]:
    """
    Graph RAG에서만 발견된 논문의 paper_id를 이용해
    PostgreSQL에서 실제 관련 Chunk를 가져온다.
    """

    graph_only_ids = [
        paper["paper_id"]
        for paper in hybrid_papers
        if (
            "graph" in paper["retrieval_sources"]
            and "vector"
            not in paper["retrieval_sources"]
        )
    ]

    if not graph_only_ids:
        return hybrid_papers

    chunk_results = search_paper_chunks_by_ids(
        query_en=query_en,
        paper_ids=graph_only_ids,
        top_k_per_paper=top_k_per_paper,
    )

    chunks_by_paper: dict[int, list[dict]] = {}

    for chunk in chunk_results:
        paper_id = chunk["paper_id"]

        chunks_by_paper.setdefault(
            paper_id,
            [],
        ).append(
            {
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "similarity": chunk["similarity"],
            }
        )

    for paper in hybrid_papers:
        paper_id = paper["paper_id"]

        if paper_id in chunks_by_paper:
            paper["evidence_chunks"] = (
                chunks_by_paper[paper_id]
            )

            # Graph 결과에는 없었던 PostgreSQL 메타데이터도 보충
            first_chunk = next(
                (
                    chunk
                    for chunk in chunk_results
                    if chunk["paper_id"] == paper_id
                ),
                None,
            )

            if first_chunk:
                paper["title"] = (
                    first_chunk.get("title")
                    or paper.get("title")
                )
                paper["authors"] = first_chunk.get(
                    "authors"
                )
                paper["published_year"] = (
                    first_chunk.get("published_year")
                    or paper.get("published_year")
                )
                paper["source"] = first_chunk.get(
                    "source"
                )
                paper["source_url"] = (
                    first_chunk.get("source_url")
                )
                paper["doi"] = first_chunk.get("doi")

    return hybrid_papers