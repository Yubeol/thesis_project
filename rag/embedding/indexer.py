import os

from pgvector import Vector

from rag.embedding.embedder import (
    embed_texts,
)
from rag.embedding.chunker import (
    chunk_news,
    chunk_paper,
)
from rag.vector.retriever import (
    get_connection,
)


EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def index_papers(
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """
    papers 테이블의 논문을 읽어:

    papers
    → chunk
    → embedding
    → paper_chunks

    force=False:
        아직 chunk가 없는 논문만 처리

    force=True:
        기존 chunk를 지우고 다시 생성
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            if force:
                sql = """
                    SELECT
                        paper_id,
                        abstract,
                        introduction,
                        body,
                        conclusion
                    FROM papers
                    ORDER BY paper_id
                """
            else:
                sql = """
                    SELECT
                        p.paper_id,
                        p.abstract,
                        p.introduction,
                        p.body,
                        p.conclusion
                    FROM papers AS p
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM paper_chunks AS pc
                        WHERE pc.paper_id = p.paper_id
                    )
                    ORDER BY p.paper_id
                """

            if limit is not None:
                sql += " LIMIT %s"
                cur.execute(
                    sql,
                    (limit,),
                )
            else:
                cur.execute(sql)

            rows = cur.fetchall()

        indexed_papers = 0
        indexed_chunks = 0

        for row in rows:
            paper = {
                "paper_id": row[0],
                "abstract": row[1],
                "introduction": row[2],
                "body": row[3],
                "conclusion": row[4],
            }

            chunks = chunk_paper(
                paper
            )

            if not chunks:
                continue

            texts = [
                chunk["content_en"]
                for chunk in chunks
            ]

            embeddings = embed_texts(
                texts
            )

            with conn.cursor() as cur:

                if force:
                    cur.execute(
                        """
                        DELETE FROM paper_chunks
                        WHERE paper_id = %s
                        """,
                        (paper["paper_id"],),
                    )

                rows_to_insert = []

                for chunk, embedding in zip(
                    chunks,
                    embeddings,
                ):
                    rows_to_insert.append(
                        (
                            paper["paper_id"],
                            chunk["section"],
                            chunk["chunk_index"],
                            chunk["content_en"],
                            Vector(embedding),
                            EMBEDDING_MODEL,
                            chunk["token_count"],
                        )
                    )

                cur.executemany(
                    """
                    INSERT INTO paper_chunks (
                        paper_id,
                        section,
                        chunk_index,
                        content_en,
                        embedding,
                        embedding_model,
                        token_count
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (
                        paper_id,
                        chunk_index
                    )
                    DO UPDATE SET
                        section = EXCLUDED.section,
                        content_en = EXCLUDED.content_en,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        token_count = EXCLUDED.token_count
                    """,
                    rows_to_insert,
                )

            indexed_papers += 1
            indexed_chunks += len(chunks)

        conn.commit()

    return {
        "indexed_papers": indexed_papers,
        "indexed_chunks": indexed_chunks,
    }


def index_news(
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """
    news 테이블의 영어 RAG용 본문을 읽어:

    news
    → chunk
    → embedding
    → news_chunks
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            if force:
                sql = """
                    SELECT
                        news_id,
                        content_en_for_rag
                    FROM news
                    ORDER BY news_id
                """
            else:
                sql = """
                    SELECT
                        n.news_id,
                        n.content_en_for_rag
                    FROM news AS n
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM news_chunks AS nc
                        WHERE nc.news_id = n.news_id
                    )
                    ORDER BY n.news_id
                """

            if limit is not None:
                sql += " LIMIT %s"
                cur.execute(
                    sql,
                    (limit,),
                )
            else:
                cur.execute(sql)

            rows = cur.fetchall()

        indexed_news = 0
        indexed_chunks = 0

        for row in rows:
            news = {
                "news_id": row[0],
                "content_en_for_rag": (
                    row[1]
                ),
            }

            chunks = chunk_news(
                news
            )

            if not chunks:
                continue

            texts = [
                chunk["content_en"]
                for chunk in chunks
            ]

            embeddings = embed_texts(
                texts
            )

            with conn.cursor() as cur:

                if force:
                    cur.execute(
                        """
                        DELETE FROM news_chunks
                        WHERE news_id = %s
                        """,
                        (news["news_id"],),
                    )

                rows_to_insert = []

                for chunk, embedding in zip(
                    chunks,
                    embeddings,
                ):
                    rows_to_insert.append(
                        (
                            news["news_id"],
                            chunk["chunk_index"],
                            chunk["content_en"],
                            Vector(embedding),
                            EMBEDDING_MODEL,
                            chunk["token_count"],
                        )
                    )

                cur.executemany(
                    """
                    INSERT INTO news_chunks (
                        news_id,
                        chunk_index,
                        content_en,
                        embedding,
                        embedding_model,
                        token_count
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (
                        news_id,
                        chunk_index
                    )
                    DO UPDATE SET
                        content_en = EXCLUDED.content_en,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        token_count = EXCLUDED.token_count
                    """,
                    rows_to_insert,
                )

            indexed_news += 1
            indexed_chunks += len(chunks)

        conn.commit()

    return {
        "indexed_news": indexed_news,
        "indexed_chunks": indexed_chunks,
    }