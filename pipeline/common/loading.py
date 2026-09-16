"""Transactional insert-only loading into existing papers/news, with read-only preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit

import psycopg
from psycopg import sql

from pipeline.common.database import DEFAULT_ENV, DatabaseConfigurationError, connect, inspect_table

TABLES = {
    "papers": {"pk": "paper_id", "identity": ["doi", "source_url", "content_hash"],
               "types": {"title": "character varying(500)", "authors": "text", "published_year": "integer",
                         "abstract": "text", "introduction": "text", "body": "text", "conclusion": "text",
                         "keywords": "text[]", "source": "character varying(200)", "source_url": "text",
                         "doi": "character varying(255)", "language": "character varying(10)", "content_hash": "character varying(64)"}},
    "news": {"pk": "news_id", "identity": ["url", "content_hash"],
             "types": {"original_language": "character varying(10)", "title_original": "character varying(500)",
                       "title_en_for_rag": "character varying(500)", "content_original": "text", "content_en_for_rag": "text",
                       "published_at": "timestamp without time zone", "source": "character varying(200)", "url": "text",
                       "category": "character varying(100)", "keywords": "text[]", "content_hash": "character varying(64)",
                       "collected_at": "timestamp without time zone"}},
}


class LoadValidationError(ValueError):
    pass


def content_hash(text: str) -> str:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise LoadValidationError("Timestamp must include its timezone before loading")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def validate_values(table: str, row: dict) -> None:
    for name, kind in TABLES[table]["types"].items():
        value = row[name]
        if value is None:
            continue
        match = re.fullmatch(r"character varying\((\d+)\)", kind)
        if kind == "text" or match:
            if not isinstance(value, str) or "\x00" in value:
                raise LoadValidationError(name + " must be text without NUL characters")
            if match and len(value) > int(match[1]):
                raise LoadValidationError(name + " exceeds database length limit")
        if kind == "text[]" and (not isinstance(value, list) or any(not isinstance(x, str) or "\x00" in x for x in value)):
            raise LoadValidationError(name + " must be a list of text values")
        if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise LoadValidationError(name + " must be an integer")


def prepare_record(table: str, record: dict) -> dict:
    row = {name: record.get(name) for name in TABLES[table]["types"]}
    if table == "papers":
        if not row["title"] or not row["source_url"]:
            raise LoadValidationError("Paper title and source_url are required")
        if record.get("quality_state") not in (None, "ready"):
            raise LoadValidationError("Paper is not ready for database loading")
        if {"source_title_mismatch", "text_encoding_damage", "extraction_failed"} & set(record.get("quality_flags", [])):
            raise LoadValidationError("Paper requires source/extraction review before loading")
        if isinstance(row["authors"], list):
            row["authors"] = json.dumps(row["authors"], ensure_ascii=False)
        if row["doi"]:
            row["doi"] = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", unquote(row["doi"]).strip(), flags=re.I).lower()
        identity_text = record.get("fulltext") or "\n".join(row.get(k) or "" for k in ("title", "abstract", "introduction", "body", "conclusion"))
        row["content_hash"] = content_hash(identity_text)
        source_url = row["source_url"]
    else:
        if not row["title_original"] or not row["content_original"] or not row["url"]:
            raise LoadValidationError("News requires original title, full article text and source URL")
        if record.get("content_status") != "fulltext_extracted":
            raise LoadValidationError("News descriptions/metadata are not full articles")
        if not row["title_en_for_rag"] or not row["content_en_for_rag"]:
            raise LoadValidationError("English RAG fields are not ready")
        if row["original_language"] not in {"ko", "en"}:
            raise LoadValidationError("News language is not supported by the current pipeline")
        row["published_at"] = utc_timestamp(row["published_at"])
        row["collected_at"] = utc_timestamp(row["collected_at"])
        if not row["published_at"]:
            raise LoadValidationError("News publication time is required")
        row["content_hash"] = content_hash(row["content_original"])
        source_url = row["url"]
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise LoadValidationError("Expected HTTP(S) source URL without credentials")
    row["keywords"] = row["keywords"] or []
    validate_values(table, row)
    return row


def validate_schema(table: str, schema: dict) -> None:
    columns = schema["columns"]
    for name, expected in {TABLES[table]["pk"]: "bigint", **TABLES[table]["types"]}.items():
        if name not in columns or columns[name]["type"] != expected:
            raise LoadValidationError("Existing schema mismatch at " + table + "." + name)
    pk = TABLES[table]["pk"]
    if ("p", "PRIMARY KEY (" + pk + ")") not in schema["constraints"]:
        raise LoadValidationError("Expected primary key is missing")
    if not columns[pk]["default"]:
        raise LoadValidationError("Database must allocate the primary key")
    for name, info in columns.items():
        if name not in TABLES[table]["types"] and name != pk and info["not_null"] and info["default"] is None:
            raise LoadValidationError("Unmapped required column: " + table + "." + name)
    if not schema["can_select"]:
        raise LoadValidationError("SELECT permission is required")


def find_existing(cursor, table: str, row: dict) -> list[int]:
    keys = [key for key in TABLES[table]["identity"] if row.get(key)]
    expressions = []
    values = []
    for key in keys:
        # DOI matching is case-insensitive for existing rows written by other clients.
        expressions.append(sql.SQL("lower({})=lower(%s)").format(sql.Identifier(key)) if key == "doi"
                           else sql.SQL("{}=%s").format(sql.Identifier(key)))
        values.append(row[key])
    query = sql.SQL("SELECT {} FROM public.{} WHERE ").format(sql.Identifier(TABLES[table]["pk"]), sql.Identifier(table))
    cursor.execute(query + sql.SQL(" OR ").join(expressions), values)
    return sorted({item[0] for item in cursor.fetchall()})


def load(table: str, records: list[dict], env_file: Path, *, apply: bool = False) -> dict:
    if table not in TABLES:
        raise LoadValidationError("Table is outside the papers/news loader scope")
    prepared, rejected = [], []
    for record in records:
        try:
            prepared.append((record["id"], prepare_record(table, record)))
        except (LoadValidationError, TypeError, KeyError, ValueError) as exc:
            rejected.append({"source_id": record.get("id"), "reason": str(exc)})
    report = {"table": table, "mode": "apply" if apply else "read_only_preflight", "input_count": len(records),
              "prepared_count": len(prepared), "inserted": 0, "existing": 0, "would_insert": 0,
              "rejected": rejected, "id_mappings": [], "committed": False}
    with connect(env_file, read_only=not apply) as conn:
        with conn.cursor() as cursor:
            schema = inspect_table(cursor, table)
            validate_schema(table, schema)
            report["can_insert"] = schema["can_insert"]
            if apply:
                if not schema["can_insert"]:
                    raise LoadValidationError("INSERT permission is required")
                # Existing schema has no unique DOI/hash/URL constraint. Serialize this short batch
                # against all other writers, including clients not using advisory locks.
                cursor.execute(sql.SQL("LOCK TABLE public.{} IN SHARE ROW EXCLUSIVE MODE").format(sql.Identifier(table)))
            cursor.execute(sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table)))
            report["count_before"] = cursor.fetchone()[0]
            in_batch = {}
            for source_id, row in prepared:
                keys = [(key, row[key]) for key in TABLES[table]["identity"] if row.get(key)]
                existing_ids = find_existing(cursor, table, row)
                if len(existing_ids) > 1:
                    raise LoadValidationError("Conflicting existing identities for source " + str(source_id))
                if existing_ids:
                    database_id = existing_ids[0]
                    report["existing"] += 1
                    action = "existing"
                elif any(key in in_batch for key in keys):
                    database_id = next(in_batch[key] for key in keys if key in in_batch)
                    report["existing"] += 1
                    action = "batch_duplicate"
                elif apply:
                    names = list(row)
                    query = sql.SQL("INSERT INTO public.{} ({}) VALUES ({}) RETURNING {}").format(
                        sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, names)),
                        sql.SQL(",").join(sql.Placeholder() for _ in names), sql.Identifier(TABLES[table]["pk"]))
                    cursor.execute(query, [row[name] for name in names])
                    database_id = cursor.fetchone()[0]
                    report["inserted"] += 1
                    action = "inserted"
                else:
                    database_id = None
                    report["would_insert"] += 1
                    action = "would_insert"
                for key in keys:
                    in_batch[key] = database_id
                report["id_mappings"].append({"source_id": source_id, TABLES[table]["pk"]: database_id, "action": action})
            cursor.execute(sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table)))
            report["count_after"] = cursor.fetchone()[0]
    report["committed"] = apply
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main(table: str, default_input: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=default_input or Path("data/processed") / table / (table + ".json"))
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--apply", action="store_true", help="Commit inserts; default performs SELECT-only preflight")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        report = load(table, records, args.env_file, apply=args.apply)
    except (psycopg.Error, DatabaseConfigurationError, LoadValidationError, OSError, ValueError) as exc:
        # Do not print SQL row values or libpq diagnostics containing credentials.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "sqlstate": getattr(exc, "sqlstate", None)}))
        return 1
    report_path = args.report or args.input.with_name("load_report.json" if args.apply else "load_preflight.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp = report_path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(report_path)
    print(json.dumps({key: value for key, value in report.items() if key not in {"id_mappings", "rejected"}}, ensure_ascii=True, indent=2))
    print("Rejected records:", len(report["rejected"]))
    return 0 if report["prepared_count"] and not report["rejected"] else 1
