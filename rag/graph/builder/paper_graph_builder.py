from rag.graph.graph_rag.retriever import (
    get_database_name,
    get_neo4j_driver,
)


def ensure_graph_constraints() -> None:
    """
    Neo4j 중복 노드 생성을 방지하기 위한
    Unique Constraint를 생성한다.
    """

    driver = get_neo4j_driver()
    database = get_database_name()

    queries = [
        """
        CREATE CONSTRAINT paper_id_unique
        IF NOT EXISTS
        FOR (p:Paper)
        REQUIRE p.paper_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT author_name_unique
        IF NOT EXISTS
        FOR (a:Author)
        REQUIRE a.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT keyword_name_unique
        IF NOT EXISTS
        FOR (k:Keyword)
        REQUIRE k.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT source_name_unique
        IF NOT EXISTS
        FOR (s:Source)
        REQUIRE s.name IS UNIQUE
        """,
    ]

    for query in queries:
        driver.execute_query(
            query,
            database_=database,
        )


def build_paper_graph(
    paper: dict,
) -> None:
    """
    논문 1건의 Neo4j Graph를 생성한다.

    생성:
    Paper
    Author
    Keyword
    Source

    관계:
    Author -[:WRITTEN_BY]-> Paper
    Paper -[:HAS_KEYWORD]-> Keyword
    Paper -[:PUBLISHED_BY]-> Source
    """

    driver = get_neo4j_driver()
    database = get_database_name()

    driver.execute_query(
        """
        MERGE (p:Paper {
            paper_id: $paper_id
        })

        SET
            p.title = $title,
            p.published_year = $published_year,
            p.abstract_summary = $abstract
        """,
        paper_id=paper["paper_id"],
        title=paper["title"],
        published_year=paper["published_year"],
        abstract=paper["abstract"],
        database_=database,
    )

    for author in paper["authors"]:
        driver.execute_query(
            """
            MATCH (p:Paper {
                paper_id: $paper_id
            })

            MERGE (a:Author {
                name: $author
            })

            MERGE (a)-[:WRITTEN_BY]->(p)
            """,
            paper_id=paper["paper_id"],
            author=author,
            database_=database,
        )

    for keyword in paper["keywords"]:
        driver.execute_query(
            """
            MATCH (p:Paper {
                paper_id: $paper_id
            })

            MERGE (k:Keyword {
                name: $keyword
            })

            MERGE (p)-[:HAS_KEYWORD]->(k)
            """,
            paper_id=paper["paper_id"],
            keyword=keyword,
            database_=database,
        )

    if paper["source"]:
        driver.execute_query(
            """
            MATCH (p:Paper {
                paper_id: $paper_id
            })

            MERGE (s:Source {
                name: $source
            })

            MERGE (p)-[:PUBLISHED_BY]->(s)
            """,
            paper_id=paper["paper_id"],
            source=paper["source"],
            database_=database,
        )