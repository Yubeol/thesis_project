"""Collect traceable K-pop paper metadata; never invent missing paper sections."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import unquote, urlsplit

from pipeline.common.http import FetchError, get_json

LOG = logging.getLogger(__name__)
CONFIG = Path(__file__).with_name("config.json")
CORE = re.compile(r"\bk[\s\-–]?pop\b|\bkorean (?:pop|wave)\b|\bhallyu\b|케이팝|한류", re.I)
CONTEXT = {
    "fandom": r"fan(?:dom|s|ning)?\b|팬덤|팬\s",
    "social_media": r"social media|sns\b|tiktok|youtube|twitter|instagram|소셜|소셜미디어",
    "diffusion": r"global|transnational|international|diffusion|circulation|해외|세계|글로벌",
    "participation": r"participat|communit|activit|consum|팬활동|커뮤니티",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def normalize_title(title: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", title).casefold() if c.isalnum())


def normalize_doi(doi: str | None) -> str | None:
    value = unquote(doi or "").strip().lower()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def abstract_text(index: dict | None) -> str | None:
    if not index:
        return None
    positions = {}
    for word, offsets in index.items():
        for position in offsets:
            positions[int(position)] = word
    return " ".join(positions[k] for k in sorted(positions)) or None


def relevant(title: str, abstract: str | None) -> tuple[int, list[str]]:
    combined = title + " " + (abstract or "")
    if not CORE.search(combined):
        return 0, []
    matches = [name for name, pattern in CONTEXT.items() if re.search(pattern, combined, re.I)]
    if not matches:
        return 0, []
    score = 5 * bool(CORE.search(title)) + len(matches)
    score += sum(bool(re.search(pattern, title, re.I)) for pattern in CONTEXT.values())
    return score, matches


def normalize_work(work: dict, query: str) -> dict | None:
    title = (work.get("title") or "").strip()
    abstract = abstract_text(work.get("abstract_inverted_index"))
    score, matches = relevant(title, abstract)
    if not title or not score or work.get("is_retracted"):
        return None
    if work.get("type") != "article":
        return None
    year = work.get("publication_year")
    if not isinstance(year, int) or year > datetime.now(timezone.utc).year:
        return None
    work_id = work.get("id") or ""
    if not re.fullmatch(r"https://openalex.org/W\d+", work_id):
        return None
    doi = normalize_doi(work.get("doi"))
    locations = work.get("locations") or []
    best = work.get("best_oa_location") or {}
    locations = [best] + locations
    sources = []
    seen = set()
    for location in locations:
        if not location.get("is_oa"):
            continue
        for kind in ("pdf_url", "landing_page_url"):
            url = location.get(kind)
            if url and url not in seen and urlsplit(url).scheme in {"http", "https"}:
                seen.add(url)
                sources.append({"url": url, "kind": kind, "license": location.get("license"),
                                "version": location.get("version")})
    return {
        "id": work_id.rsplit("/", 1)[-1], "title": title,
        "authors": [a["author"]["display_name"] for a in work.get("authorships", [])
                    if (a.get("author") or {}).get("display_name")],
        "published_year": year, "abstract": abstract,
        "introduction": None, "body": None, "conclusion": None,
        "keywords": [k["display_name"] for k in work.get("keywords", []) if k.get("display_name")],
        "source": "OpenAlex", "source_url": "https://doi.org/" + doi if doi else work_id,
        "metadata_url": work_id, "doi": doi, "language": work.get("language"),
        "is_oa": bool((work.get("open_access") or {}).get("is_oa")),
        "oa_status": (work.get("open_access") or {}).get("oa_status"),
        "source_locations": sources, "matched_topics": matches,
        "relevance_score": score, "matched_queries": [query],
        "fulltext_status": "not_acquired", "section_status": "not_extracted",
    }


def select_unique(records: list[dict], target: int) -> list[dict]:
    ranked = sorted(records, key=lambda r: (-int(r["is_oa"]), -r["relevance_score"], r["id"]))
    result, identifiers = [], {}
    for record in ranked:
        keys = [("id", record["id"]), ("title", normalize_title(record["title"]))]
        if record["doi"]:
            keys.append(("doi", record["doi"]))
        previous = next((identifiers[k] for k in keys if k in identifiers), None)
        if previous is not None:
            existing = result[previous]
            existing["matched_queries"] = sorted(set(existing["matched_queries"] + record["matched_queries"]))
            known_urls = {location["url"] for location in existing["source_locations"]}
            existing["source_locations"] = existing["source_locations"] + [
                location for location in record["source_locations"] if location["url"] not in known_urls]
            if record["id"] != existing["id"]:
                existing["duplicate_metadata_ids"] = sorted(set(
                    existing.get("duplicate_metadata_ids", []) + [record["id"]]))
            for key in keys:
                identifiers[key] = previous
            continue
        for key in keys:
            identifiers[key] = len(result)
        result.append({**record, "source_locations": list(record["source_locations"])})
    return result[:target]


def collect(config: dict, output: Path, *, refresh: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    raw_count = cached_pages = fetched_pages = 0
    headers = {}
    if os.getenv("OPENALEX_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENALEX_API_KEY"]
    for query in config["queries"]:
        cursor = "*"
        for page in range(config["pages_per_query"]):
            params = {"search": query, "filter": "type:article,is_retracted:false",
                      "per_page": config["page_size"], "cursor": cursor}
            if config["open_access_only"]:
                params["filter"] += ",is_oa:true"
            cache_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
            cache = output / "api" / (cache_id + ".json")
            try:
                if cache.exists() and not refresh:
                    response = json.loads(cache.read_text(encoding="utf-8"))
                    cached_pages += 1
                else:
                    response = get_json("https://api.openalex.org/works", params, headers=headers)
                    if not isinstance(response.get("results"), list):
                        raise FetchError("OpenAlex response is missing results")
                    write_json(cache, response)
                    fetched_pages += 1
                    time.sleep(1)
                works = response["results"]
                raw_count += len(works)
                for work in works:
                    record = normalize_work(work, query)
                    if record and (not config["open_access_only"] or record["is_oa"]):
                        record["raw_metadata_path"] = cache.relative_to(output).as_posix()
                        records.append(record)
                LOG.info("Query %r page %s: %s results", query, page + 1, len(works))
                next_cursor = (response.get("meta") or {}).get("next_cursor")
                if len(works) < config["page_size"] or not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
            except (FetchError, ValueError, KeyError, TypeError) as exc:
                errors.append({"query": query, "page": page + 1, "error": str(exc)})
                LOG.error("Query failed %r: %s", query, exc)
                break
    selected = select_unique(records, config["target"])
    report = {
        "schema_version": 1, "collected_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "raw_results": raw_count, "relevant_results_before_dedup": len(records),
        "unique_candidates": len(select_unique(records, len(records))),
        "selected_count": len(selected), "with_abstract": sum(bool(r["abstract"]) for r in selected),
        "with_source_location": sum(bool(r["source_locations"]) for r in selected),
        "cached_pages": cached_pages, "fetched_pages": fetched_pages,
        "metadata_minimum_met": len(selected) >= config["minimum"],
        "fulltext_verified_count": 0, "training_ready_count": 0,
        "errors": errors,
    }
    write_json(output / "papers.json", selected)
    write_json(output / "collection_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=Path("data/raw/papers"))
    parser.add_argument("--target", type=int)
    parser.add_argument("--minimum", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("target", "minimum"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if not 1 <= config["minimum"] <= config["target"] <= 1000:
        parser.error("Require 1 <= minimum <= target <= 1000")
    if not 1 <= config["page_size"] <= 200 or not 1 <= config["pages_per_query"] <= 20:
        parser.error("page_size must be 1..200 and pages_per_query 1..20")
    if not config["queries"] or not all(isinstance(q, str) and q.strip() for q in config["queries"]):
        parser.error("Provide non-empty search queries")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "collection.log", encoding="utf-8")])
    report = collect(config, args.output, refresh=args.refresh)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["metadata_minimum_met"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
