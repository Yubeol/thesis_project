from pgvector import Vector

from rag.embedding.embedder import embed_query
from rag.vector.retriever import get_connection


DEFAULT_TOP_K = 5


def search_news_vector(
    query_en: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    영문 Query와 의미적으로 가까운
    뉴스 Chunk Top-K를 검색한다.
    """

    if not query_en or not query_en.strip():
        raise ValueError("검색 Query가 비어 있습니다.")

    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    query_embedding = Vector(
        embed_query(query_en)
    )

    sql = """
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

            1 - (nc.embedding <=> %s) AS similarity

        FROM news_chunks AS nc

        JOIN news AS n
            ON n.news_id = nc.news_id

        WHERE nc.embedding IS NOT NULL

        ORDER BY nc.embedding <=> %s

        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_embedding,
                    query_embedding,
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
        }
        for row in rows
    ]