"""Collect recent publisher RSS candidates. Descriptions are never used as full articles."""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import argparse
import hashlib
import html
import json
import logging
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

from pipeline.common.http import FetchError, fetch
from pipeline.papers.collector.openalex import write_json
from pipeline.news.deduplicator.records import canonical_url, deduplicate, news_id

LOG = logging.getLogger(__name__)
CONTEXT = {
    "social_media": r"social media|tiktok|youtube|twitter|instagram|\bsns\b|소셜|틱톡|유튜브|트위터|인스타",
    "global_market": r"billboard|global|worldwide|international|overseas|spotify|oricon|빌보드|글로벌|해외|세계|스포티파이|오리콘",
    "fan_activity": r"\bfans?\b|fandom|fanclub|fan club|fan community|팬덤|팬클럽|팬[\s들미팅]",
    "performance_consumption": r"world tour|concert|festival|album sales|million|월드투어|월드 투어|콘서트|페스티벌|판매량",
}
KPOP = re.compile(r"\bk[ -]?pop\b|케이팝|케이 팝|한류|아이돌|걸그룹|보이그룹|방탄소년단|블랙핑크|세븐틴|스트레이\s?키즈|스키즈|빅뱅|\bbts\b|\bblackpink\b|\bstray kids\b|\btwice\b|\bseventeen\b", re.I)


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def topics(text: str, *, kpop_source: bool) -> list[str]:
    if not kpop_source and not KPOP.search(text):
        return []
    return [name for name, pattern in CONTEXT.items() if re.search(pattern, text, re.I)]


def parse_feed(payload: bytes, source: dict, feed_url: str, now: datetime, lookback_days: int,
               *, stats: dict | None = None) -> list[dict]:
    root = ET.fromstring(payload)
    if root.tag.lower() != "rss":
        raise ValueError("Expected RSS document")
    records = []
    stats = stats if stats is not None else {}
    stats.update({"items": 0, "invalid": 0, "outside_date_window": 0, "outside_topics": 0, "accepted": 0})
    for item in root.findall("./channel/item"):
        stats["items"] += 1
        try:
            title = plain(item.findtext("title") or "")
            url = canonical_url(item.findtext("link") or "")
            # Some publishers use ISO-style offsets inside an RFC 822 date.
            date = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", (item.findtext("pubDate") or "").strip())
            published = parsedate_to_datetime(date)
            if published.tzinfo is None:
                raise ValueError("RSS publication time requires a timezone")
            published = published.astimezone(timezone.utc)
            if not now - timedelta(days=lookback_days) <= published <= now:
                stats["outside_date_window"] += 1
                continue
            summary = plain(item.findtext("description") or "")
            categories = [plain(node.text or "") for node in item.findall("category")]
            required_categories = source.get("require_any_category", [])
            if required_categories and not set(required_categories).intersection(categories):
                stats["outside_topics"] += 1
                continue
            content_html = item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")
            rss_fulltext = bool(source.get("rss_content_is_full") and content_html)
            matched = topics(title + " " + summary + " " + " ".join(categories)
                             + (" " + plain(content_html) if rss_fulltext else ""),
                             kpop_source=source["kpop_source"])
            if not title or not matched:
                stats["outside_topics"] += 1
                continue
            records.append({
                "id": news_id(url), "original_language": source["language"],
                "title_original": title, "content_original": None,
                "title_en_for_rag": None, "content_en_for_rag": None,
                "published_at": published.isoformat(), "source": source["name"], "url": url,
                "category": matched[0], "keywords": matched, "publisher_categories": categories,
                "rss_description": summary,
                "rss_content_html": content_html, "rss_content_is_full": rss_fulltext,
                "discovered_via": [feed_url], "kpop_source": source["kpop_source"],
                "collected_at": now.isoformat(), "content_status": "not_acquired",
                "translation_status": "not_processed",
            })
            stats["accepted"] += 1
        except (ValueError, TypeError, OverflowError, IndexError):
            stats["invalid"] += 1
            LOG.warning("Skipped invalid RSS item from %s", source["name"])
    return records


def collect(config: dict, output: Path, *, refresh=False, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    output.mkdir(parents=True, exist_ok=True)
    records, errors, source_counts, feed_reports = [], [], {}, []
    for source in config["feeds"]:
        source_counts.setdefault(source["name"], 0)
        for query in source.get("queries", [None]):
            previous_page_ids = None
            for page in range(1, source.get("pages", 1) + 1):
                params = {"s": query, "paged": page} if query else {}
                url = source["url"] + ("?" + urlencode(params) if params else "")
                key = hashlib.sha256(url.encode()).hexdigest()[:24]
                cache = output / "feeds" / now.strftime("%Y-%m-%d") / (key + ".xml")
                try:
                    if cache.exists() and not refresh:
                        payload = cache.read_bytes()
                    else:
                        payload, _, _ = fetch(url, timeout=20, max_bytes=8_000_000)
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_bytes(payload)
                        time.sleep(1)
                    stats = {"source": source["name"], "feed": url}
                    items = parse_feed(payload, source, url, now, config["lookback_days"], stats=stats)
                    feed_reports.append(stats)
                    page_ids = {item["id"] for item in items}
                    if not items or page_ids == previous_page_ids:
                        break
                    previous_page_ids = page_ids
                    for item in items:
                        item["raw_feed_path"] = cache.relative_to(output).as_posix()
                    records.extend(items)
                    source_counts[source["name"]] = source_counts.get(source["name"], 0) + len(items)
                    LOG.info("%s / %s / page %s: %s recent candidates", source["name"], query or "feed", page, len(items))
                except (FetchError, ValueError, ET.ParseError) as exc:
                    errors.append({"source": source["name"], "feed": url, "error": str(exc)})
                    LOG.error("Feed failed: %s / %s", source["name"], type(exc).__name__)
                    break
    # Keep the most recent copy of existing items and accumulate previously discovered recent URLs.
    manifest = output / "news.json"
    existing_retained_count = 0
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        cutoff = (now - timedelta(days=config["lookback_days"])).isoformat()
        active_sources = {source["name"]: source for source in config["feeds"]}
        fresh_by_url = {row["url"]: row for row in records}
        for row in old:
            source = active_sources.get(row["source"])
            if not source or not cutoff <= row["published_at"] <= now.isoformat():
                continue
            required = source.get("require_any_category", [])
            if required and not set(required).intersection(row.get("publisher_categories", [])):
                continue
            if row["url"] in fresh_by_url:
                fresh = fresh_by_url[row["url"]]
                fresh["discovered_via"] = sorted(set(fresh["discovered_via"] + row.get("discovered_via", [])))
                fresh["duplicate_sources"] = row.get("duplicate_sources", [])
                continue
            records.append(row)
            existing_retained_count += 1
    # Interleave publishers so one high-volume source does not displace all Korean stories.
    records.sort(key=lambda row: row["published_at"], reverse=True)
    unique, duplicates = deduplicate(records)
    groups = {}
    for row in unique:
        groups.setdefault(row["source"], []).append(row)
    selected = []
    while len(selected) < config["target"] and any(groups.values()):
        for group in groups.values():
            if group and len(selected) < config["target"]:
                selected.append(group.pop(0))
    report = {"collected_at": now.isoformat(), "config": config, "source_candidate_counts": source_counts,
              "feed_reports": feed_reports, "existing_retained_count": existing_retained_count,
              "selected_count": len(selected), "unique_candidates": len(unique),
              "duplicates": len(duplicates), "errors": errors,
              "metadata_minimum_met": len(selected) >= config["minimum"], "full_articles": 0}
    write_json(manifest, selected)
    write_json(output / "duplicates.json", duplicates)
    write_json(output / "collection_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/news"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not 1 <= config["minimum"] <= config["target"] <= 1000 or not 1 <= config["lookback_days"] <= 3660:
            raise ValueError("Invalid target/minimum/lookback_days")
        if not isinstance(config["feeds"], list) or not 1 <= len(config["feeds"]) <= 20:
            raise ValueError("Require 1..20 configured feeds")
        for source in config["feeds"]:
            canonical_url(source["url"])
            if not isinstance(source["name"], str) or not source["name"].strip():
                raise ValueError("Feeds require a source name")
            if source["language"] not in {"ko", "en"} or not isinstance(source["kpop_source"], bool):
                raise ValueError("Feeds require ko/en language and boolean kpop_source")
            if not 1 <= source.get("pages", 1) <= 10 or len(source.get("queries", [None])) > 20:
                raise ValueError("Feeds are limited to 10 pages and 20 queries")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"Invalid news configuration: {exc}")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "collection.log", encoding="utf-8")])
    try:
        report = collect(config, args.output, refresh=args.refresh)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.error("Collection stopped: %s", exc)
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["metadata_minimum_met"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
