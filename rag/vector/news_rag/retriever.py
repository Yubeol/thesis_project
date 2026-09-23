from pgvector import Vector

from rag.embedding.embedder import embed_query
from rag.vector.retriever import get_connection


DEFAULT_TOP_K = 5
NEWS_CANDIDATE_MULTIPLIER = 4


def search_news_vector(
    query_en: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    영문 Query와 의미적으로 가까운 뉴스들을 검색한다.

    같은 뉴스에 여러 Chunk가 검색될 경우,
    뉴스별 similarity가 가장 높은 Chunk 1개만 반환한다.

    먼저 의미 유사도가 높은 후보군을 고른 뒤 최근성에 작은 가산점을 준다.
    따라서 관련성이 낮은 최신 기사가 관련성이 높은 과거 기사를 무조건
    밀어내지는 않는다.
    """

    if not query_en or not query_en.strip():
        raise ValueError("검색 Query가 비어 있습니다.")

    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    query_embedding = Vector(
        embed_query(query_en)
    )

    candidate_limit = max(
        top_k,
        top_k * NEWS_CANDIDATE_MULTIPLIER,
    )

    sql = """
        WITH ranked_chunks AS (
            SELECT
                nc.chunk_id,
                nc.news_id,
                nc.chunk_index,
                nc.content_en,

                n.title_original,
                n.title_en_for_rag,
                n.original_language,
                n.published_at,
                n.source,
                n.url,

                1 - (nc.embedding <=> %s) AS similarity,

                ROW_NUMBER() OVER (
                    PARTITION BY nc.news_id
                    ORDER BY nc.embedding <=> %s
                ) AS news_rank

            FROM news_chunks AS nc

            JOIN news AS n
                ON n.news_id = nc.news_id

            WHERE nc.embedding IS NOT NULL
        ),

        best_news AS (
            SELECT
                chunk_id,
                news_id,
                chunk_index,
                content_en,
                title_original,
                title_en_for_rag,
                original_language,
                published_at,
                source,
                url,
                similarity
            FROM ranked_chunks
            WHERE news_rank = 1
            ORDER BY similarity DESC
            LIMIT %s
        ),

        candidates AS (
            SELECT
                *,
                CASE
                    WHEN published_at IS NULL THEN 0.0
                    WHEN published_at >= CURRENT_TIMESTAMP - INTERVAL '3 years'
                        THEN 1.0
                    WHEN published_at >= CURRENT_TIMESTAMP - INTERVAL '5 years'
                        THEN 0.5
                    ELSE 0.0
                END AS recency_score
            FROM best_news
        )

        SELECT
            chunk_id,
            news_id,
            chunk_index,
            content_en,

            title_original,
            title_en_for_rag,
            original_language,
            published_at,
            source,
            url,

            similarity,
            recency_score,
            similarity + (recency_score * 0.08) AS ranking_score

        FROM candidates

        ORDER BY ranking_score DESC, similarity DESC

        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_embedding,
                    query_embedding,
                    candidate_limit,
                    top_k,
                ),
            )

            rows = cur.fetchall()

    return [
        {
            "chunk_id": row[0],
            "news_id": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "title_original": row[4],
            "title_en": row[5],
            "original_language": row[6],
            "published_at": row[7],
            "source": row[8],
            "url": row[9],
            "similarity": float(row[10]),
            "recency_score": float(row[11]),
            "ranking_score": float(row[12]),
        }
        for row in rows
    ]
