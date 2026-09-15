import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()
DEFAULT_TOP_K = 5

def get_connection():
    """
    PostgreSQL + pgvector 연결을 생성한다.
    """

    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    missing = [
        name
        for name, value in {
            "POSTGRES_HOST": host,
            "POSTGRES_DB": dbname,
            "POSTGRES_USER": user,
            "POSTGRES_PASSWORD": password,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "PostgreSQL 환경변수가 없습니다: "
            + ", ".join(missing)
        )

    conn = psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=10,
    )

    register_vector(conn)

    return conn

def search_papers_vector(
    query_en: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    사용자의 영문 Query와 의미적으로 가까운
    논문 Chunk Top-K를 검색한다.
    """

    if not query_en or not query_en.strip():
        raise ValueError("검색 Query가 비어 있습니다.")

    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    query_embedding = Vector(embed_query(query_en))

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

    results = []

    for row in rows:
        results.append(
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
        )

    return results


def search_news_vector(
    query_en: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    사용자의 영문 Query와 의미적으로 가까운
    뉴스 Chunk Top-K를 검색한다.
    """

    if not query_en or not query_en.strip():
        raise ValueError("검색 Query가 비어 있습니다.")

    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    query_embedding = Vector(embed_query(query_en))

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

    results = []

    for row in rows:
        results.append(
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
        )

    return results

def search_paper_chunks_by_ids(
    query_en: str,
    paper_ids: list[int],
    top_k_per_paper: int = 2,
) -> list[dict]:
    """
    Graph RAG에서 발견한 paper_id들을 대상으로
    PostgreSQL의 실제 논문 Chunk를 다시 검색한다.

    각 논문 안에서 사용자의 Query와 가장 유사한
    Chunk를 top_k_per_paper개씩 가져온다.
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

    results = []

    for row in rows:
        results.append(
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
        )

    return results