"""Connection and schema inspection for the existing EC2 PostgreSQL; never runs DDL."""

from pathlib import Path
import os

import psycopg
from dotenv import dotenv_values

DEFAULT_ENV = Path(__file__).resolve().parents[2] / ".env"


class DatabaseConfigurationError(ValueError):
    pass


def connect(env_file: Path = DEFAULT_ENV, *, read_only: bool = True):
    file_values = dotenv_values(env_file) if env_file.is_file() else {}
    def value(name):
        return os.environ.get(name) or file_values.get(name)
    required = ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [name for name in required if not value(name)]
    if missing:
        raise DatabaseConfigurationError("Missing connection variables: " + ", ".join(missing))
    try:
        port = int(value("POSTGRES_PORT") or "5432")
        if not 1 <= port <= 65535:
            raise ValueError()
    except ValueError:
        raise DatabaseConfigurationError("POSTGRES_PORT must be 1..65535") from None
    kwargs = dict(host=value("POSTGRES_HOST"), port=port, dbname=value("POSTGRES_DB"),
                  user=value("POSTGRES_USER"), password=value("POSTGRES_PASSWORD"),
                  connect_timeout=10, application_name="suam_pipeline",
                  options="-c default_transaction_read_only=" + ("on" if read_only else "off")
                          + " -c statement_timeout=30000 -c lock_timeout=10000 -c timezone=UTC")
    if value("POSTGRES_SSLMODE"):
        kwargs["sslmode"] = value("POSTGRES_SSLMODE")
    try:
        return psycopg.connect(**kwargs)
    except psycopg.Error as exc:
        # libpq exception strings can contain connection data. Return classification only.
        raise DatabaseConfigurationError("PostgreSQL connection failed: " + type(exc).__name__) from None


def inspect_table(cursor, table: str) -> dict:
    if table not in {"papers", "news", "training_samples"}:
        raise ValueError("Table is outside Suam's write scope")
    cursor.execute("""SELECT a.attname, format_type(a.atttypid,a.atttypmod), a.attnotnull,
                           pg_get_expr(d.adbin,d.adrelid)
                      FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
                      JOIN pg_namespace n ON n.oid=c.relnamespace
                      LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
                     WHERE n.nspname='public' AND c.relname=%s
                       AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum""", (table,))
    columns = {name: {"type": kind, "not_null": required, "default": default}
               for name, kind, required, default in cursor.fetchall()}
    cursor.execute("SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid=to_regclass(%s)", ("public." + table,))
    constraints = cursor.fetchall()
    cursor.execute("SELECT has_table_privilege(current_user,%s,'SELECT'), has_table_privilege(current_user,%s,'INSERT')", ("public." + table, "public." + table))
    select_ok, insert_ok = cursor.fetchone()
    return {"columns": columns, "constraints": constraints, "can_select": select_ok, "can_insert": insert_ok}
