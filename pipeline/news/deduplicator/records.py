"""Stable news identity with provenance-preserving URL/title/content deduplication."""

import hashlib
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonical_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected public HTTP(S) news URL without embedded credentials")
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", urlencode(sorted(query)), ""))


def normalized_text(text: str) -> str:
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", text).casefold())


def news_id(url: str) -> str:
    return "news_" + hashlib.sha256(canonical_url(url).encode()).hexdigest()[:24]


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    unique, index, duplicates = [], {}, []
    for record in records:
        keys = [("url", canonical_url(record["url"]))]
        title = normalized_text(record.get("title_original") or "")
        if title:
            keys.append(("title", title))
        content = normalized_text(record.get("content_original") or "")
        if content:
            keys.append(("content", hashlib.sha256(content.encode()).hexdigest()))
        position = next((index[key] for key in keys if key in index), None)
        if position is not None:
            existing = unique[position]
            alternatives = existing.get("duplicate_sources", []) + record.get("duplicate_sources", []) + [
                {"id": record["id"], "source": record["source"], "url": record["url"]}]
            # Re-running the same feed must not grow a list of self duplicates.
            existing["duplicate_sources"] = list({(item["source"], item["url"]): item
                for item in alternatives if (item["source"], item["url"]) != (existing["source"], existing["url"])}.values())
            existing["discovered_via"] = sorted(set(existing.get("discovered_via", []) + record.get("discovered_via", [])))
            duplicates.append({"id": record["id"], "retained_id": existing["id"], "url": record["url"]})
        else:
            position = len(unique)
            unique.append(dict(record))
        for key in keys:
            index[key] = position
    return unique, duplicates
