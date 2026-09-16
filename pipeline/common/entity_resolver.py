"""Resolve source mentions against a small, source-backed alias catalogue.

The catalogue is deliberately explicit: a translation or a similar spelling is
never evidence that two named entities have the same identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit


LOG = logging.getLogger(__name__)
DEFAULT_CATALOG = Path(__file__).with_name("entity_aliases.json")
ENTITY_TYPES = frozenset({
    "Artist", "Group", "Person", "Work", "Drama", "Movie", "Album",
    "Song", "Fandom", "Platform", "Company", "Topic",
})


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _mention_pattern(alias: str) -> re.Pattern:
    # Hangul particles may immediately follow a name; Latin names still need
    # word boundaries so BTS does not match a longer unrelated token.
    before = r"(?<![A-Za-z0-9가-힣])"
    after = r"(?![A-Za-z0-9])" if re.search(r"[가-힣]$", alias) else r"(?![A-Za-z0-9가-힣])"
    return re.compile(before + re.escape(alias) + after, re.IGNORECASE)


class EntityResolver:
    def __init__(self, entries: list[dict]):
        self.entries = []
        self.by_alias: dict[str, list[dict]] = {}
        ids: set[str] = set()
        for source in entries:
            kind = source.get("entity_type")
            urls = source.get("source_urls") or []
            authority = source.get("authority_url")
            if kind not in ENTITY_TYPES or not isinstance(authority, str):
                raise ValueError("Catalogue entry needs a supported type and authority URL")
            if urlsplit(authority).scheme != "https" or not urlsplit(authority).hostname:
                raise ValueError("Catalogue authority must be an HTTPS URL")
            if not urls or any(urlsplit(url).scheme != "https" for url in urls):
                raise ValueError("Catalogue entry needs HTTPS evidence URLs")
            names = [source.get("name_ko"), source.get("name_en")]
            aliases = list(dict.fromkeys(x for x in [*names, *(source.get("aliases") or [])] if isinstance(x, str) and x.strip()))
            if not aliases:
                raise ValueError("Catalogue entry has no name or alias")
            entity_id = kind.lower() + "_" + hashlib.sha256(authority.encode("utf-8")).hexdigest()[:24]
            if entity_id in ids:
                raise ValueError("Duplicate catalogue authority and type")
            ids.add(entity_id)
            item = {**source, "entity_id": entity_id, "aliases": aliases}
            self.entries.append(item)
            for alias in aliases:
                self.by_alias.setdefault(normalize(alias), []).append(item)

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CATALOG) -> "EntityResolver":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Entity catalogue must be a JSON list")
        return cls(payload)

    def resolve(self, surface: str, language: str, entity_type: str | None = None,
                context: str = "", translated_form: str | None = None) -> dict:
        surface = surface.strip()
        if not surface:
            raise ValueError("Empty entity surface")
        candidates = self.by_alias.get(normalize(surface), [])
        if entity_type:
            candidates = [item for item in candidates if item["entity_type"] == entity_type]
        candidates = [item for item in candidates if not item.get("context_terms") or any(
            re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", context, re.I)
            for term in item["context_terms"])]
        if len(candidates) == 1:
            item = candidates[0]
            LOG.info("ENTITY_ALIAS_MATCH entity_id=%s", item["entity_id"])
            LOG.info("ENTITY_RESOLVED entity_id=%s type=%s", item["entity_id"], item["entity_type"])
            return {
                "entity_id": item["entity_id"], "entity_type": item["entity_type"],
                "surface_form": surface, "name_ko": item.get("name_ko"),
                "name_en": item.get("name_en"),
                "canonical_name": item.get("name_en") or item.get("name_ko"),
                "aliases": item["aliases"], "resolution_status": "resolved",
                "confidence": 1.0, "source_urls": item["source_urls"],
            }
        status = "translated_only" if translated_form else "unresolved"
        LOG.info("ENTITY_UNRESOLVED type=%s", entity_type or "unknown")
        return {
            "entity_id": None, "entity_type": entity_type,
            "surface_form": surface, "name_ko": surface if language == "ko" else None,
            "name_en": None, "canonical_name": None, "aliases": [surface],
            "resolution_status": status, "confidence": 0.0,
            "candidate_translation": translated_form, "source_urls": [],
        }

    def resolve_document(self, *, title: str, text: str, language: str,
                         existing_entities: list[dict | str] | None = None) -> list[dict]:
        context = title + "\n" + text
        requests: list[tuple[str, str | None, str | None]] = []
        for item in self.entries:
            for alias in item["aliases"]:
                match = _mention_pattern(alias).search(context)
                if match:
                    # An untyped text hit is ambiguous when an alias belongs
                    # to more than one entity or entity type.
                    requests.append((match.group(), None, None))
        if language == "ko":
            # Quoted Korean titles are candidates, not asserted identities.
            # Preserve them as unresolved unless a verified alias matches.
            for match in re.finditer(r"[‘“〈《]([^’”〉》\n]{2,40})[’”〉》]", title + "\n" + text[:2000]):
                candidate = match.group(1).strip()
                if re.search(r"[가-힣]", candidate):
                    requests.append((candidate, None, None))
        for item in existing_entities or []:
            if isinstance(item, str):
                requests.append((item, None, None))
            elif isinstance(item, dict):
                surface = item.get("surface_form") or item.get("text") or item.get("name")
                if surface:
                    requests.append((surface, item.get("entity_type"), item.get("translated_form")))
        resolved = []
        seen = set()
        for surface, kind, translated in requests:
            key = (normalize(surface), kind)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(self.resolve(surface, language, kind, context, translated))
        return resolved


def resolve_record(record: dict, kind: str, resolver: EntityResolver) -> dict:
    result = dict(record)
    title = record.get("title_original" if kind == "news" else "title") or ""
    text = record.get("content_original" if kind == "news" else "fulltext") or ""
    language = record.get("original_language" if kind == "news" else "language") or ""
    try:
        result["entities"] = resolver.resolve_document(
            title=title, text=text, language=language,
            existing_entities=record.get("entities"),
        )
        result["entity_resolution_status"] = "completed"
    except Exception as exc:
        # The source record is preserved even when resolution fails.
        LOG.warning("ENTITY_UNRESOLVED resolver_error=%s", type(exc).__name__)
        unresolved = []
        for item in record.get("entities", []) or []:
            surface = item if isinstance(item, str) else (
                item.get("surface_form") or item.get("text") if isinstance(item, dict) else None
            )
            if not surface:
                continue
            unresolved.append({
                "entity_id": None,
                "entity_type": item.get("entity_type") if isinstance(item, dict) else None,
                "surface_form": surface,
                "name_ko": surface if language == "ko" else None,
                "name_en": None, "canonical_name": None,
                "aliases": [surface], "resolution_status": "unresolved",
                "confidence": 0.0, "source_urls": [],
            })
        result["entities"] = unresolved
        result["entity_resolution_status"] = "failed"
    return result


def protect_official_names(title: str, content: str,
                           entities: list[dict]) -> tuple[str, str, dict[str, str]]:
    """Replace only verified names with opaque markers before translation."""
    replacements: dict[str, str] = {}
    seen = set()
    for entity in sorted(entities, key=lambda x: len(x.get("surface_form") or ""), reverse=True):
        if entity.get("resolution_status") != "resolved" or not entity.get("name_en"):
            continue
        surface = entity.get("surface_form") or ""
        if not surface or normalize(surface) in seen:
            continue
        seen.add(normalize(surface))
        marker = f"ENTITYPROTECTTOKEN{len(replacements):04d}"
        pattern = _mention_pattern(surface)
        title, title_count = pattern.subn(marker, title)
        content, content_count = pattern.subn(marker, content)
        if title_count + content_count:
            replacements[marker] = entity["name_en"]
            LOG.info("ENTITY_TRANSLATION_PROTECTED entity_id=%s", entity["entity_id"])
    return title, content, replacements


def restore_official_names(title: str, content: str, markers: dict[str, str],
                           source_title: str, source_content: str) -> tuple[str, str]:
    for marker, name in markers.items():
        if title.count(marker) != source_title.count(marker) or content.count(marker) != source_content.count(marker):
            raise ValueError("Translation did not preserve a verified entity marker")
        title = title.replace(marker, name)
        content = content.replace(marker, name)
    return title, content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["news", "papers"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    source = args.input or Path("data/processed") / args.kind / (args.kind + ".json")
    target = args.output or source.with_name(source.stem + "_resolved.json")
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(x, dict) for x in records):
        parser.error("Input must be a list of JSON records")
    try:
        resolver = EntityResolver.from_file(args.catalog)
    except (OSError, ValueError, TypeError) as exc:
        LOG.warning("ENTITY_UNRESOLVED catalogue_error=%s", type(exc).__name__)
        resolver = EntityResolver([])
    ready = [resolve_record(record, args.kind, resolver) for record in records]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"input_count": len(records), "resolved_mentions": sum(
        x.get("resolution_status") == "resolved" for row in ready for x in row["entities"]),
        "unresolved_mentions": sum(x.get("resolution_status") != "resolved"
        for row in ready for x in row["entities"]), "output": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
