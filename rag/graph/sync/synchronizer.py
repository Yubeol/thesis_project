from rag.graph.builder.paper_graph_builder import (
    build_paper_graph,
    ensure_graph_constraints,
)
from rag.vector.retriever import get_connection


def _normalize_authors(
    authors: str | None,
) -> list[str]:
    if not authors:
        return []

    authors = authors.strip()

    if not authors:
        return []

    if ";" in authors:
        return [
            author.strip()
            for author in authors.split(";")
            if author.strip()
        ]

    return [authors]


def _normalize_keywords(
    keywords,
) -> list[str]:
    if not keywords:
        return []

    result = []
    seen = set()

    for keyword in keywords:
        if not keyword:
            continue

        keyword = keyword.strip()

        if not keyword:
            continue

        key = keyword.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(keyword)

    return result


def fetch_unsynced_papers(
    limit: int | None = None,
) -> list[dict]:

    sql = """
        SELECT
            paper_id,
            title,
            authors,
            published_year,
            abstract,
            keywords,
            source
        FROM papers
        WHERE graph_synced = FALSE
        ORDER BY paper_id
    """

    params = ()

    if limit is not None:
        sql += " LIMIT %s"
        params = (limit,)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "paper_id": row[0],
            "title": row[1],
            "authors": _normalize_authors(row[2]),
            "published_year": row[3],
            "abstract": row[4],
            "keywords": _normalize_keywords(row[5]),
            "source": (
                row[6].strip()
                if row[6]
                else None
            ),
        }
        for row in rows
    ]


def mark_paper_synced(
    paper_id: int,
) -> None:

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE papers
                SET graph_synced = TRUE
                WHERE paper_id = %s
                """,
                (paper_id,),
            )

        conn.commit()


def sync_papers_to_graph(
    limit: int | None = None,
) -> dict:

    ensure_graph_constraints()

    papers = fetch_unsynced_papers(
        limit=limit
    )

    synced = 0
    failed = []

    for paper in papers:
        try:
            build_paper_graph(paper)

            mark_paper_synced(
                paper["paper_id"]
            )

            synced += 1

        except Exception as exc:
            failed.append(
                {
                    "paper_id": paper["paper_id"],
                    "error": str(exc),
                }
            )

    return {
        "requested": len(papers),
        "synced": synced,
        "failed": failed,
    }