"""Keep the existing paper graph and attach resolved mentions from both sources."""

from __future__ import annotations

import argparse
import json
import logging

from pipeline.common.entity_resolver import EntityResolver, resolve_record
from rag.graph.builder.paper_graph_builder import (
    build_paper_graph,
    ensure_graph_constraints,
)
from rag.vector.retriever import get_connection
from rag.graph.sync.entity_sync import (
    ensure_entity_constraints,
    merge_entity_mentions,
    merge_news_node,
)


LOG = logging.getLogger(__name__)


def _resolver_or_empty() -> EntityResolver:
    try:
        return EntityResolver.from_file()
    except (OSError, ValueError, TypeError) as exc:
        LOG.warning("ENTITY_UNRESOLVED catalogue_error=%s", type(exc).__name__)
        return EntityResolver([])


def _normalize_authors(
    authors: str | None,
) -> list[str]:
    if not authors:
        return []

    if authors.startswith("["):
        try:
            values = json.loads(authors)
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                return [value.strip() for value in values if value.strip()]
        except json.JSONDecodeError:
            pass

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
    *,
    reconcile: bool = False,
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
            , introduction
            , body
            , conclusion
            , language
        FROM papers
        WHERE (%s OR graph_synced = FALSE)
        ORDER BY paper_id
    """

    params = (reconcile,)

    if limit is not None:
        sql += " LIMIT %s"
        params = (reconcile, limit)

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
            "introduction": row[7],
            "body": row[8],
            "conclusion": row[9],
            "language": row[10],
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
    *,
    reconcile: bool = False,
) -> dict:

    ensure_graph_constraints()
    ensure_entity_constraints()
    resolver = _resolver_or_empty()

    papers = fetch_unsynced_papers(
        limit=limit, reconcile=reconcile
    )

    synced = 0
    failed = []

    for paper in papers:
        try:
            build_paper_graph(paper)

            resolved = resolve_record({
                "title": paper["title"],
                "fulltext": "\n".join(paper.get(part) or "" for part in
                                     ("abstract", "introduction", "body", "conclusion")),
                "language": paper.get("language"),
            }, "papers", resolver)
            merge_entity_mentions("Paper", paper["paper_id"], resolved["entities"])

            mark_paper_synced(
                paper["paper_id"]
            )

            synced += 1

        except Exception as exc:
            failed.append(
                {
                    "paper_id": paper["paper_id"],
                    "error_type": type(exc).__name__,
                }
            )

    return {
        "requested": len(papers),
        "synced": synced,
        "failed": failed,
    }

def ensure_news_graph_sync_column() -> None:
    """
    News Graph 증분 동기화를 위한 상태 컬럼을 보장한다.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                ALTER TABLE news
                ADD COLUMN IF NOT EXISTS graph_synced
                BOOLEAN NOT NULL DEFAULT FALSE
                """
            )

        conn.commit()


def fetch_unsynced_news(
    limit: int | None = None,
    *,
    reconcile: bool = False,
) -> list[dict]:

    query = """
        SELECT
            news_id,
            original_language,
            title_original,
            content_original,
            published_at,
            source,
            url
        FROM news
        WHERE (%s OR graph_synced = FALSE)
        ORDER BY news_id
    """

    params = (reconcile,)

    if limit is not None:
        query += " LIMIT %s"
        params = (reconcile, limit)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    return [
        dict(
            zip(
                (
                    "news_id",
                    "original_language",
                    "title_original",
                    "content_original",
                    "published_at",
                    "source",
                    "url",
                ),
                row,
            )
        )
        for row in rows
    ]

def mark_news_synced(
    news_id: int,
) -> None:

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE news
                SET graph_synced = TRUE
                WHERE news_id = %s
                """,
                (news_id,),
            )

        conn.commit()

def sync_news_to_graph(
    limit: int | None = None,
    *,
    reconcile: bool = False,
) -> dict:

    ensure_news_graph_sync_column()
    ensure_entity_constraints()

    resolver = _resolver_or_empty()

    news = fetch_unsynced_news(
        limit=limit,
        reconcile=reconcile,
    )

    synced = 0
    mentions = 0
    failed = []

    for record in news:
        try:
            merge_news_node(record)

            resolved = resolve_record(
                record,
                "news",
                resolver,
            )

            mentions += merge_entity_mentions(
                "News",
                record["news_id"],
                resolved["entities"],
            )

            mark_news_synced(
                record["news_id"]
            )

            synced += 1

        except Exception as exc:
            LOG.warning(
                "News graph sync failed news_id=%s error=%s",
                record["news_id"],
                type(exc).__name__,
            )

            failed.append(
                {
                    "news_id": record["news_id"],
                    "error_type": type(exc).__name__,
                }
            )

    return {
        "requested": len(news),
        "synced": synced,
        "mentions": mentions,
        "failed": failed,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["papers", "news", "both"], default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reconcile-papers", action="store_true",
                        help="Recheck already synced papers after catalogue updates")
    parser.add_argument(
        "--reconcile-news",
        action="store_true",
        help="Recheck already synced news after catalogue updates",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    reports = {}
    if args.source in {"papers", "both"}:
        reports["papers"] = sync_papers_to_graph(args.limit, reconcile=args.reconcile_papers)
    if args.source in {"news", "both"}:
        reports["news"] = sync_news_to_graph(
            args.limit,
            reconcile=args.reconcile_news,
        )
    print(json.dumps(reports, ensure_ascii=False))
    return 0 if all(not report["failed"] for report in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
