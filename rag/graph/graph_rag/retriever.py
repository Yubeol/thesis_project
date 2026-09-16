import os
from functools import lru_cache

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


@lru_cache
def get_neo4j_driver():
    """
    Neo4j Driver를 생성한다.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri:
        raise RuntimeError("NEO4J_URI가 설정되어 있지 않습니다.")

    if not user:
        raise RuntimeError("NEO4J_USER가 설정되어 있지 않습니다.")

    if not password:
        raise RuntimeError("NEO4J_PASSWORD가 설정되어 있지 않습니다.")

    return GraphDatabase.driver(
        uri,
        auth=(user, password),
    )


def get_database_name() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


def verify_graph_connection() -> bool:
    """
    Neo4j 연결 상태를 확인한다.
    """
    driver = get_neo4j_driver()
    driver.verify_connectivity()

    return True


def search_related_papers(
    paper_ids: list[int],
    limit: int = 10,
) -> list[dict]:
    """
    Vector RAG에서 검색된 paper_id를 기준으로
    Neo4j에서 관계가 있는 논문을 확장 검색한다.

    검색 관계:
    - CITES
    - RELATED_TO
    - 공통 Topic
    - 공통 Keyword
    """

    if not paper_ids:
        return []

    if limit < 1:
        raise ValueError("limit은 1 이상이어야 합니다.")

    driver = get_neo4j_driver()
    database = get_database_name()

    query = """
    MATCH (seed:Paper)
    WHERE seed.paper_id IN $paper_ids

    MATCH path = (seed)-[
        r:CITES|RELATED_TO|HAS_TOPIC|HAS_KEYWORD
    *1..2]-(related:Paper)

    WHERE NOT related.paper_id IN $paper_ids

    WITH
        related,
        count(path) AS relation_score,
        collect(DISTINCT [rel IN relationships(path) | type(rel)])
            AS relation_paths

    RETURN
        related.paper_id AS paper_id,
        related.title AS title,
        related.published_year AS published_year,
        relation_score,
        relation_paths

    ORDER BY relation_score DESC

    LIMIT $limit
    """

    records, _, _ = driver.execute_query(
        query,
        paper_ids=paper_ids,
        limit=limit,
        database_=database,
    )

    return [
        {
            "paper_id": record["paper_id"],
            "title": record["title"],
            "published_year": record["published_year"],
            "relation_score": record["relation_score"],
            "relation_paths": record["relation_paths"],
        }
        for record in records
    ]


def get_paper_graph_context(
    paper_id: int,
) -> dict | None:
    """
    특정 논문의 Graph 정보를 조회한다.

    Author
    Topic
    Keyword
    Source
    Institution
    """

    driver = get_neo4j_driver()
    database = get_database_name()

    query = """
    MATCH (p:Paper {paper_id: $paper_id})

    OPTIONAL MATCH (a:Author)-[:WRITTEN_BY]->(p)
    OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
    OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
    OPTIONAL MATCH (p)-[:PUBLISHED_BY]->(s:Source)
    OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:Institution)

    RETURN
        p.paper_id AS paper_id,
        p.title AS title,

        collect(DISTINCT a.name) AS authors,
        collect(DISTINCT t.name) AS topics,
        collect(DISTINCT k.name) AS keywords,
        collect(DISTINCT s.name) AS sources,
        collect(DISTINCT i.name) AS institutions
    """

    records, _, _ = driver.execute_query(
        query,
        paper_id=paper_id,
        database_=database,
    )

    if not records:
        return None

    record = records[0]

    return {
        "paper_id": record["paper_id"],
        "title": record["title"],
        "authors": record["authors"],
        "topics": record["topics"],
        "keywords": record["keywords"],
        "sources": record["sources"],
        "institutions": record["institutions"],
    }