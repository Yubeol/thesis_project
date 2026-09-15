import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


load_dotenv()


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