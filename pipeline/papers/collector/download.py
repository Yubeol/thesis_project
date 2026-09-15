"""Acquire OA source documents. Downloaded HTML is not yet verified full text."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import threading
import time
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from pipeline.common.http import FetchError, USER_AGENT, fetch
from pipeline.papers.collector.openalex import write_json

LOG = logging.getLogger(__name__)


class SourceLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdfs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name", "").lower() == "citation_pdf_url":
            if attrs.get("content"):
                self.pdfs.append(attrs["content"])
        # OJS and repository landing pages can expose their own PDF through links
        # or an embedded viewer without citation_pdf_url metadata.
        url = attrs.get("href") if tag == "a" else attrs.get("src") if tag in {"embed", "iframe"} else None
        if url and (attrs.get("type") == "application/pdf" or re.search(r"\.pdf(?:[?#]|$)|/article/download/", url, re.I)):
            if url not in self.pdfs:
                self.pdfs.append(url)


class AccessPolicy:
    """Serialize each origin, respect robots.txt, fail closed on policy failures."""

    def __init__(self):
        self._guard = threading.Lock()
        self._locks = {}
        self._robots = {}

    def read(self, url: str):
        parsed = urlsplit(url)
        origin = parsed.scheme + "://" + parsed.netloc
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise FetchError("Invalid source URL")
        with self._guard:
            lock = self._locks.setdefault(origin, threading.Lock())
        with lock:
            if origin not in self._robots:
                robot = RobotFileParser(origin + "/robots.txt")
                try:
                    payload, _, _ = fetch(robot.url, attempts=2, timeout=10, max_bytes=500_000)
                    robot.parse(payload.decode("utf-8", errors="replace").splitlines())
                    self._robots[origin] = robot
                except FetchError as exc:
                    # A missing robots.txt permits access; all other failures defer acquisition.
                    if "HTTP 404 " in str(exc):
                        robot.parse([])
                        self._robots[origin] = robot
                    else:
                        self._robots[origin] = None
            robot = self._robots[origin]
            if robot is None or not robot.can_fetch(USER_AGENT, url):
                raise FetchError(f"Source access deferred by robots policy at {parsed.hostname}")
            delay = robot.crawl_delay(USER_AGENT) or robot.crawl_delay("*") or 1
            if delay > 30:
                raise FetchError(f"Source crawl delay exceeds run budget at {parsed.hostname}")
            time.sleep(delay)
            return fetch(url, attempts=2, timeout=15, max_bytes=20_000_000)


def source_format(payload: bytes, content_type: str) -> str:
    if payload.startswith(b"%PDF-"):
        return "pdf"
    if content_type in {"text/html", "application/xhtml+xml"}:
        if len(payload) < 500:
            raise FetchError("HTML response too short to be a source document")
        text = payload[:100_000].decode("utf-8", errors="replace").lower()
        if any(marker in text for marker in (
            "<title>just a moment", "<title>access denied", "cf-chl-", "verify you are human",
            "<title>attention required", "<title>robot or human",
        )):
            raise FetchError("Source returned an access challenge")
        return "html"
    raise FetchError("Response is not a PDF or HTML source document")


def download_one(paper: dict, root: Path, policy: AccessPolicy) -> dict:
    identifier = paper["id"]
    if not re.fullmatch(r"W\d+", identifier):
        raise ValueError("Invalid paper identifier")
    directory = root / "documents" / identifier
    manifest = directory / "source.json"
    if manifest.exists():
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        filename = cached.get("filename", "")
        if filename in {"source.pdf", "source.html"}:
            local = directory / filename
            if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == cached.get("sha256"):
                return {**cached, "cache_hit": True}
    failures = []
    locations = sorted(paper["source_locations"], key=lambda loc: loc["kind"] != "pdf_url")
    fallback = None
    for location in locations[:4]:
        url = location["url"]
        try:
            payload, content_type, resolved = policy.read(url)
            kind = source_format(payload, content_type)
            if kind == "pdf":
                chosen = (payload, kind, resolved, location)
                break
            if fallback is None:
                fallback = (payload, kind, resolved, location)
            links = SourceLinks()
            links.feed(payload.decode("utf-8", errors="replace"))
            chosen = None
            for link in links.pdfs[:3]:
                pdf_url = urljoin(resolved, link)
                try:
                    pdf, pdf_type, pdf_resolved = policy.read(pdf_url)
                    if source_format(pdf, pdf_type) != "pdf":
                        raise FetchError("PDF metadata link returned HTML")
                    chosen = (pdf, "pdf", pdf_resolved, location)
                    break
                except FetchError as exc:
                    failures.append({"url": pdf_url, "error": str(exc)})
            if chosen:
                break
        except FetchError as exc:
            failures.append({"url": url, "error": str(exc)})
    else:
        chosen = fallback
    if not chosen:
        result = {"id": identifier, "status": "unavailable", "attempts": failures}
        LOG.warning("%s: no source acquired (%s failed attempts)", identifier, len(failures))
        return result
    payload, kind, resolved, location = chosen
    directory.mkdir(parents=True, exist_ok=True)
    filename = "source." + kind
    temp = directory / (filename + ".tmp")
    temp.write_bytes(payload)
    temp.replace(directory / filename)
    result = {
        "id": identifier, "status": "downloaded_unverified", "filename": filename,
        "format": kind, "source_url": location["url"], "resolved_url": resolved,
        "license": location.get("license"), "license_source": "OpenAlex location metadata",
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "attempts": failures,
    }
    write_json(manifest, result)
    LOG.info("%s: downloaded %s (%s bytes)", identifier, kind, len(payload))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/papers"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    papers = json.loads((args.root / "papers.json").read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.root / "download.log", encoding="utf-8")])
    policy = AccessPolicy()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda paper: download_one(paper, args.root, policy), papers))
    report = {
        "finished_at": datetime.now(timezone.utc).isoformat(), "selected_count": len(papers),
        "downloaded_pdf": sum(r.get("format") == "pdf" for r in results),
        "downloaded_html": sum(r.get("format") == "html" for r in results),
        "unavailable": sum(r["status"] == "unavailable" for r in results),
        "fulltext_verified_count": 0, "training_ready_count": 0, "documents": results,
    }
    write_json(args.root / "download_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, indent=2))
    return 0 if report["downloaded_pdf"] + report["downloaded_html"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
