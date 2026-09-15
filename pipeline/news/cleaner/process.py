"""Acquire publisher article text; keep Korean original and pending translation separate."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re

from trafilatura import bare_extraction
from pipeline.common.http import FetchError
from pipeline.papers.collector.download import AccessPolicy
from pipeline.papers.collector.openalex import write_json
from pipeline.news.collector.rss import topics
from pipeline.news.deduplicator.records import deduplicate, normalized_text

LOG = logging.getLogger(__name__)


def clean_article_text(text: str, row: dict) -> str:
    """Remove duplicated headings and identified publisher promotional footers."""
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if lines and normalized_text(lines[0]) == normalized_text(row["title_original"]):
        lines.pop(0)
    if row["source"] == "Soompi":
        for index, line in enumerate(lines):
            if re.match(r"(?:watch|in the meantime, watch|meanwhile, watch|or (?:watch|check out))\b.*\bViki\b.*below", line, re.I):
                lines = lines[:index]
                break
    return "\n".join(lines)


def process_one(row: dict, root: Path, policy: AccessPolicy) -> dict:
    if not re.fullmatch(r"news_[0-9a-f]{24}", row["id"]):
        raise ValueError("Invalid news ID")
    directory = root / "articles" / row["id"]
    directory.mkdir(parents=True, exist_ok=True)
    document_path = directory / "source.html"
    manifest_path = directory / "source.json"
    manifest = None
    if row.get("rss_content_html") and row.get("rss_content_is_full"):
        payload = row["rss_content_html"].encode("utf-8")
        resolved = row["url"]
        method = "publisher_rss_content"
    elif document_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = document_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            raise ValueError("Cached news source checksum mismatch")
        resolved, method = manifest["resolved_url"], manifest["method"]
    else:
        payload, content_type, resolved = policy.read(row["url"])
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise FetchError("News URL did not return HTML")
        method = "publisher_html"
    # RSS content:encoded is an HTML fragment, which trafilatura can discard
    # without a document/body wrapper. Only explicitly configured full feeds
    # qualify; ordinary descriptions and unverified encoded snippets do not.
    extraction_input = payload
    if method == "publisher_rss_content":
        extraction_input = '<html><head><meta charset="utf-8"></head><body><article>' + payload.decode("utf-8") + '</article></body></html>'
    article = bare_extraction(extraction_input, include_comments=False, include_tables=True, with_metadata=True)
    text = clean_article_text(article.text, row) if article and article.text else None
    if not text or len(text) < 200:
        raise ValueError("Publisher full text unavailable or shorter than 200 characters")
    matched = topics(row["title_original"] + " " + text, kpop_source=row["kpop_source"])
    if not matched:
        raise ValueError("Article is outside K-pop research topics")
    # Title mismatch frequently means a redirect to a homepage or an access page.
    title_words = set(re.findall(r"\w{3,}", row["title_original"].casefold()))
    text_words = set(re.findall(r"\w{3,}", ((article.title or "") + " " + text).casefold()))
    if title_words and len(title_words & text_words) / len(title_words) < 0.5:
        raise ValueError("Publisher text does not match RSS title")
    document_path.write_bytes(payload)
    manifest = manifest or {"url": row["url"], "resolved_url": resolved, "method": method,
                           "sha256": hashlib.sha256(payload).hexdigest(),
                           "retrieved_at": row["collected_at"] if method == "publisher_rss_content" else datetime.now(timezone.utc).isoformat()}
    write_json(manifest_path, manifest)
    english = row["original_language"] == "en"
    return {**row, "content_original": text, "content_status": "fulltext_extracted",
            "title_en_for_rag": row["title_original"] if english else None,
            "content_en_for_rag": text if english else None,
            "translation_status": "not_required" if english else "pending",
            "keywords": matched, "category": matched[0], "source_document": manifest,
            "content_sha256": hashlib.sha256(normalized_text(text).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/news"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/news"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.minimum < 1:
        parser.error("Require 1..4 workers and a positive minimum")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "processing.log", encoding="utf-8")])
    try:
        rows = json.loads((args.root / "news.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("News manifest must contain at least one candidate")
    except (OSError, ValueError) as exc:
        LOG.error("Cannot load news manifest: %s", exc)
        return 1
    policy = AccessPolicy()
    def execute(row):
        try:
            record = process_one(row, args.root, policy)
            LOG.info("%s: %s chars, language=%s", row["id"], len(record["content_original"]), row["original_language"])
            return record, None
        except (FetchError, ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            identity = row.get("id", "invalid") if isinstance(row, dict) else "invalid"
            url = row.get("url") if isinstance(row, dict) else None
            LOG.warning("%s: %s", identity, exc)
            return None, {"id": identity, "url": url, "error": str(exc)}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(execute, rows))
    records, duplicates = deduplicate([record for record, error in results if record])
    report = {"finished_at": datetime.now(timezone.utc).isoformat(), "candidate_count": len(rows),
              "full_articles": len(records), "duplicates": len(duplicates),
              "korean_originals": sum(r["original_language"] == "ko" for r in records),
              "english_rag_ready": sum(bool(r["content_en_for_rag"]) for r in records),
              "pending_translation": sum(r["translation_status"] == "pending" for r in records),
              "source_counts": {source: sum(r["source"] == source for r in records)
                                for source in sorted({r["source"] for r in records})},
              "topic_counts": {topic: sum(topic in r["keywords"] for r in records)
                               for topic in sorted({topic for r in records for topic in r["keywords"]})},
              "publication_range": {"earliest": min((r["published_at"] for r in records), default=None),
                                    "latest": max((r["published_at"] for r in records), default=None)},
              "failed_count": sum(bool(error) for record, error in results),
              "minimum_met": len(records) >= args.minimum,
              "failures": [error for record, error in results if error]}
    write_json(args.output / "news.json", records)
    write_json(args.output / "duplicates.json", duplicates)
    write_json(args.output / "processing_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    return 0 if report["minimum_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
