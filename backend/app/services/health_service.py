from rag.graph import verify_graph_connection
from rag.vector import get_connection


def check_dependencies() -> dict:
    postgres_ok = False
    neo4j_ok = False

    postgres_error = None
    neo4j_error = None

    # PostgreSQL
    try:
        with get_connection() as conn:
            result = conn.execute(
                "SELECT 1"
            ).fetchone()

            postgres_ok = (
                result is not None
                and result[0] == 1
            )

    except Exception as exc:
        postgres_error = str(exc)

    # Neo4j
    try:
        neo4j_ok = verify_graph_connection()

    except Exception as exc:
        neo4j_error = str(exc)

    ready = postgres_ok and neo4j_ok

    return {
        "ready": ready,
        "postgres": {
            "connected": postgres_ok,
            "error": postgres_error,
        },
        "neo4j": {
            "connected": neo4j_ok,
            "error": neo4j_error,
        },
    }