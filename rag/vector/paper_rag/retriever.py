from pgvector import Vector

from rag.embedding.embedder import embed_query
from rag.vector.retriever import get_connection


DEFAULT_TOP_K = 5


def search_papers_vector(
    query_en: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    영문 Query와 의미적으로 가까운
    논문 Chunk Top-K를 검색한다.
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
            pc.chunk_id,
            pc.paper_id,
            pc.section,
            pc.chunk_index,
            pc.content_en,

            p.title,
            p.authors,
            p.published_year,
            p.source,
            p.source_url,
            p.doi,

            1 - (pc.embedding <=> %s) AS similarity

        FROM paper_chunks AS pc

        JOIN papers AS p
            ON p.paper_id = pc.paper_id

        WHERE pc.embedding IS NOT NULL

        ORDER BY pc.embedding <=> %s

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
            "paper_id": row[1],
            "section": row[2],
            "chunk_index": row[3],
            "content": row[4],
            "title": row[5],
            "authors": row[6],
            "published_year": row[7],
            "source": row[8],
            "source_url": row[9],
            "doi": row[10],
            "similarity": float(row[11]),
        }
        for row in rows
    ]


def search_paper_chunks_by_ids(
    query_en: str,
    paper_ids: list[int],
    top_k_per_paper: int = 2,
) -> list[dict]:
    """
    Graph RAG에서 발견한 paper_id들을 대상으로
    각 논문의 관련 Chunk를 다시 검색한다.
    """

    if not paper_ids:
        return []

    if not query_en or not query_en.strip():
        raise ValueError("검색 Query가 비어 있습니다.")

    if top_k_per_paper < 1:
        raise ValueError(
            "top_k_per_paper는 1 이상이어야 합니다."
        )

    query_embedding = Vector(
        embed_query(query_en)
    )

    sql = """
        WITH ranked_chunks AS (
            SELECT
                pc.chunk_id,
                pc.paper_id,
                pc.section,
                pc.chunk_index,
                pc.content_en,

                p.title,
                p.authors,
                p.published_year,
                p.source,
                p.source_url,
                p.doi,

                1 - (
                    pc.embedding <=> %s
                ) AS similarity,

                ROW_NUMBER() OVER (
                    PARTITION BY pc.paper_id
                    ORDER BY pc.embedding <=> %s
                ) AS chunk_rank

            FROM paper_chunks AS pc

            JOIN papers AS p
                ON p.paper_id = pc.paper_id

            WHERE
                pc.paper_id = ANY(%s)
                AND pc.embedding IS NOT NULL
        )

        SELECT
            chunk_id,
            paper_id,
            section,
            chunk_index,
            content_en,
            title,
            authors,
            published_year,
            source,
            source_url,
            doi,
            similarity,
            chunk_rank

        FROM ranked_chunks

        WHERE chunk_rank <= %s

        ORDER BY
            paper_id,
            chunk_rank;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_embedding,
                    query_embedding,
                    paper_ids,
                    top_k_per_paper,
                ),
            )

            rows = cur.fetchall()

    return [
        {
            "chunk_id": row[0],
            "paper_id": row[1],
            "section": row[2],
            "chunk_index": row[3],
            "content": row[4],
            "title": row[5],
            "authors": row[6],
            "published_year": row[7],
            "source": row[8],
            "source_url": row[9],
            "doi": row[10],
            "similarity": float(row[11]),
            "chunk_rank": row[12],
        }
        for row in rows
    ]