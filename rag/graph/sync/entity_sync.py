"""Idempotent Neo4j mention links using opaque, verified entity IDs."""

from __future__ import annotations

import logging

from pipeline.common.entity_resolver import ENTITY_TYPES
from rag.graph.graph_rag.retriever import get_database_name, get_neo4j_driver


LOG = logging.getLogger(__name__)


def ensure_entity_constraints() -> None:
    driver = get_neo4j_driver()
    database = get_database_name()
    for query in (
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE CONSTRAINT news_id_unique IF NOT EXISTS FOR (n:News) REQUIRE n.news_id IS UNIQUE",
    ):
        driver.execute_query(query, database_=database)


def merge_news_node(news: dict) -> None:
    driver = get_neo4j_driver()
    driver.execute_query(
        """
        MERGE (n:News {news_id: $news_id})
        SET n.title_original = $title_original,
            n.source = $source,
            n.url = $url,
            n.published_at = $published_at
        """,
        news_id=news["news_id"], title_original=news.get("title_original"),
        source=news.get("source"), url=news.get("url"),
        published_at=news["published_at"].isoformat() if news.get("published_at") else None,
        database_=get_database_name(),
    )


def merge_entity_mentions(document_label: str, document_id: int,
                          entities: list[dict]) -> int:
    if document_label not in {"Paper", "News"}:
        raise ValueError("Unsupported graph document label")
    id_property = "paper_id" if document_label == "Paper" else "news_id"
    driver = get_neo4j_driver()
    database = get_database_name()
    count = 0
    for entity in entities:
        if entity.get("resolution_status") != "resolved" or not entity.get("entity_id"):
            continue
        kind = entity["entity_type"]
        if kind not in ENTITY_TYPES:
            raise ValueError("Unsupported entity type")
        # Labels and document property are chosen only from fixed local allowlists.
        query = f"""
            MATCH (d:{document_label} {{{id_property}: $document_id}})
            MERGE (e:Entity {{entity_id: $entity_id}})
            SET e:{kind}, e.entity_type = $entity_type,
                e.name_ko = $name_ko, e.name_en = $name_en,
                e.canonical_name = $canonical_name,
                e.aliases = $aliases, e.source_urls = $source_urls
            MERGE (d)-[:MENTIONS]->(e)
        """
        driver.execute_query(
            query, document_id=document_id, entity_id=entity["entity_id"],
            entity_type=kind, name_ko=entity.get("name_ko"),
            name_en=entity.get("name_en"),
            canonical_name=entity.get("canonical_name"),
            aliases=entity.get("aliases") or [],
            source_urls=entity.get("source_urls") or [], database_=database,
        )
        LOG.info("ENTITY_RESOLVED graph_link=%s entity_id=%s", document_label, entity["entity_id"])
        count += 1
    return count
