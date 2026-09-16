"""Acquire OA source documents.

Downloads accessible PDF/HTML sources for papers collected from OpenAlex.

Important behavior:
- Individual paper download failures do not abort the whole batch.
- Transient network failures such as IncompleteRead are retried.
- A paper that still fails after retries is marked unavailable.
- The pipeline continues processing the remaining papers.
- The command fails only when no source document was acquired at all.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.client
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


# ============================================================
# Retry configuration
# ============================================================

NETWORK_RETRIES = 3

NETWORK_RETRY_BASE_DELAY_SECONDS = 1.0


# ============================================================
# HTML source link extraction
# ============================================================


class SourceLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdfs: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if (
            tag == "meta"
            and attrs.get("name", "").lower()
            == "citation_pdf_url"
        ):
            content = attrs.get("content")

            if content:
                self.pdfs.append(content)

        # OJS / repository landing pages may expose the PDF
        # through normal links or embedded viewers.
        url = (
            attrs.get("href")
            if tag == "a"
            else (
                attrs.get("src")
                if tag in {"embed", "iframe"}
                else None
            )
        )

        if (
            url
            and (
                attrs.get("type")
                == "application/pdf"
                or re.search(
                    r"\.pdf(?:[?#]|$)|/article/download/",
                    url,
                    re.I,
                )
            )
        ):
            if url not in self.pdfs:
                self.pdfs.append(url)


# ============================================================
# Network fetch wrapper
# ============================================================


def resilient_fetch(
    url: str,
    *,
    attempts: int,
    timeout: int,
    max_bytes: int,
):
    """Call the common fetch helper and recover from low-level network errors.

    pipeline.common.http.fetch already handles normal FetchError retry logic.

    Some socket/http.client errors, especially IncompleteRead, can escape
    outside FetchError. Those errors are retried here and finally converted
    to FetchError so that one bad source cannot terminate the full batch.
    """

    last_error: Exception | None = None

    for retry in range(1, NETWORK_RETRIES + 1):
        try:
            return fetch(
                url,
                attempts=attempts,
                timeout=timeout,
                max_bytes=max_bytes,
            )

        except FetchError:
            # FetchError is already normalized by the common HTTP layer.
            # Let the caller decide whether to try another source URL.
            raise

        except (http.client.HTTPException, OSError) as exc:
            last_error = exc

            LOG.warning(
                "Transient network failure "
                "(attempt %s/%s, error=%s)",
                retry,
                NETWORK_RETRIES,
                type(exc).__name__,
            )

            if retry < NETWORK_RETRIES:
                delay = (
                    NETWORK_RETRY_BASE_DELAY_SECONDS
                    * (2 ** (retry - 1))
                )

                time.sleep(delay)

    error_name = (
        type(last_error).__name__
        if last_error is not None
        else "UnknownNetworkError"
    )

    raise FetchError(
        "Network source read failed after retries: "
        + error_name
    )


# ============================================================
# Access / robots policy
# ============================================================


class AccessPolicy:
    """Serialize each origin, respect robots.txt, and fail closed.

    The lock is per origin so several different publishers can still
    be processed concurrently while requests to one publisher remain
    rate-limited.
    """

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._robots: dict[
            str,
            RobotFileParser | None,
        ] = {}

    def read(self, url: str):
        parsed = urlsplit(url)

        origin = (
            parsed.scheme
            + "://"
            + parsed.netloc
        )

        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
        ):
            raise FetchError(
                "Invalid source URL"
            )

        with self._guard:
            lock = self._locks.setdefault(
                origin,
                threading.Lock(),
            )

        with lock:
            if origin not in self._robots:
                robot = RobotFileParser(
                    origin + "/robots.txt"
                )

                try:
                    payload, _, _ = resilient_fetch(
                        robot.url,
                        attempts=2,
                        timeout=10,
                        max_bytes=500_000,
                    )

                    robot.parse(
                        payload.decode(
                            "utf-8",
                            errors="replace",
                        ).splitlines()
                    )

                    self._robots[origin] = robot

                except FetchError as exc:
                    # Missing robots.txt permits access.
                    #
                    # Other errors fail closed because we could not
                    # determine the publisher's crawling policy.
                    if "HTTP 404 " in str(exc):
                        robot.parse([])
                        self._robots[origin] = robot

                    else:
                        self._robots[origin] = None

            robot = self._robots[origin]

            if (
                robot is None
                or not robot.can_fetch(
                    USER_AGENT,
                    url,
                )
            ):
                raise FetchError(
                    "Source access deferred by "
                    "robots policy at "
                    + parsed.hostname
                )

            delay = (
                robot.crawl_delay(USER_AGENT)
                or robot.crawl_delay("*")
                or 1
            )

            if delay > 30:
                raise FetchError(
                    "Source crawl delay exceeds "
                    "run budget at "
                    + parsed.hostname
                )

            time.sleep(delay)

            return resilient_fetch(
                url,
                attempts=2,
                timeout=15,
                max_bytes=20_000_000,
            )


# ============================================================
# Source format validation
# ============================================================


def source_format(
    payload: bytes,
    content_type: str,
) -> str:
    if payload.startswith(b"%PDF-"):
        return "pdf"

    normalized_type = (
        content_type
        .split(";", 1)[0]
        .strip()
        .lower()
    )

    if normalized_type in {
        "text/html",
        "application/xhtml+xml",
    }:
        if len(payload) < 500:
            raise FetchError(
                "HTML response too short "
                "to be a source document"
            )

        text = (
            payload[:100_000]
            .decode(
                "utf-8",
                errors="replace",
            )
            .lower()
        )

        challenge_markers = (
            "<title>just a moment",
            "<title>access denied",
            "cf-chl-",
            "verify you are human",
            "<title>attention required",
            "<title>robot or human",
        )

        if any(
            marker in text
            for marker in challenge_markers
        ):
            raise FetchError(
                "Source returned an access challenge"
            )

        return "html"

    raise FetchError(
        "Response is not a PDF or HTML "
        "source document"
    )


# ============================================================
# Cached source handling
# ============================================================


def load_cached_source(
    identifier: str,
    directory: Path,
    manifest: Path,
) -> dict | None:
    """Return a valid cached source if possible.

    A damaged manifest no longer aborts the run. It is ignored and the
    paper is downloaded again.
    """

    if not manifest.exists():
        return None

    try:
        cached = json.loads(
            manifest.read_text(
                encoding="utf-8"
            )
        )

        filename = cached.get(
            "filename",
            "",
        )

        if filename not in {
            "source.pdf",
            "source.html",
        }:
            return None

        local = directory / filename

        if not local.exists():
            return None

        digest = hashlib.sha256(
            local.read_bytes()
        ).hexdigest()

        if digest != cached.get("sha256"):
            LOG.warning(
                "%s: cached source checksum mismatch; "
                "downloading again",
                identifier,
            )

            return None

        return {
            **cached,
            "cache_hit": True,
        }

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        LOG.warning(
            "%s: invalid cached source ignored "
            "(%s)",
            identifier,
            type(exc).__name__,
        )

        return None


# ============================================================
# Individual paper acquisition
# ============================================================


def download_one(
    paper: dict,
    root: Path,
    policy: AccessPolicy,
) -> dict:
    identifier = paper["id"]

    if not re.fullmatch(
        r"W\d+",
        identifier,
    ):
        raise ValueError(
            "Invalid paper identifier"
        )

    directory = (
        root
        / "documents"
        / identifier
    )

    manifest = (
        directory
        / "source.json"
    )

    cached = load_cached_source(
        identifier,
        directory,
        manifest,
    )

    if cached is not None:
        LOG.info(
            "%s: source cache hit",
            identifier,
        )

        return cached

    failures: list[dict] = []

    locations = sorted(
        paper.get(
            "source_locations",
            [],
        ),
        key=lambda loc: (
            loc.get("kind")
            != "pdf_url"
        ),
    )

    fallback = None
    chosen = None

    # Limit acquisition work per paper so one paper cannot
    # consume the entire scheduled run.
    for location in locations[:4]:
        url = location.get("url")

        if not url:
            continue

        try:
            payload, content_type, resolved = (
                policy.read(url)
            )

            kind = source_format(
                payload,
                content_type,
            )

            # Direct PDF is preferred.
            if kind == "pdf":
                chosen = (
                    payload,
                    kind,
                    resolved,
                    location,
                )

                break

            # HTML is preserved as fallback while looking
            # for an embedded/linked PDF.
            if fallback is None:
                fallback = (
                    payload,
                    kind,
                    resolved,
                    location,
                )

            links = SourceLinks()

            try:
                links.feed(
                    payload.decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            except Exception as exc:
                # HTML parser failure should not lose the HTML
                # fallback that was already successfully acquired.
                LOG.warning(
                    "%s: source HTML link parsing failed "
                    "(%s)",
                    identifier,
                    type(exc).__name__,
                )

            for link in links.pdfs[:3]:
                pdf_url = urljoin(
                    resolved,
                    link,
                )

                try:
                    (
                        pdf,
                        pdf_type,
                        pdf_resolved,
                    ) = policy.read(
                        pdf_url
                    )

                    if (
                        source_format(
                            pdf,
                            pdf_type,
                        )
                        != "pdf"
                    ):
                        raise FetchError(
                            "PDF metadata link "
                            "returned HTML"
                        )

                    chosen = (
                        pdf,
                        "pdf",
                        pdf_resolved,
                        location,
                    )

                    break

                except FetchError as exc:
                    failures.append(
                        {
                            "url": pdf_url,
                            "error": str(exc),
                        }
                    )

            if chosen is not None:
                break

        except FetchError as exc:
            failures.append(
                {
                    "url": url,
                    "error": str(exc),
                }
            )

    if chosen is None:
        chosen = fallback

    if chosen is None:
        result = {
            "id": identifier,
            "status": "unavailable",
            "failure_reason": "no_accessible_source",
            "attempts": failures,
        }

        LOG.warning(
            "%s: no source acquired "
            "(%s failed attempts)",
            identifier,
            len(failures),
        )

        return result

    payload, kind, resolved, location = (
        chosen
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "source."
        + kind
    )

    destination = (
        directory
        / filename
    )

    temp = (
        directory
        / (
            filename
            + ".tmp"
        )
    )

    try:
        temp.write_bytes(payload)
        temp.replace(destination)

    except OSError as exc:
        temp.unlink(
            missing_ok=True
        )

        raise FetchError(
            "Failed to persist downloaded "
            "source: "
            + type(exc).__name__
        ) from exc

    result = {
        "id": identifier,
        "status": "downloaded_unverified",
        "filename": filename,
        "format": kind,

        "source_url": location.get(
            "url"
        ),

        "resolved_url": resolved,

        "license": location.get(
            "license"
        ),

        "license_source":
            "OpenAlex location metadata",

        "sha256": hashlib.sha256(
            payload
        ).hexdigest(),

        "bytes": len(payload),

        "retrieved_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "attempts": failures,
    }

    write_json(
        manifest,
        result,
    )

    LOG.info(
        "%s: downloaded %s (%s bytes)",
        identifier,
        kind,
        len(payload),
    )

    return result


# ============================================================
# Per-paper failure boundary
# ============================================================


def download_one_safe(
    paper: dict,
    root: Path,
    policy: AccessPolicy,
) -> dict:
    """Prevent one broken paper/source from aborting the batch."""

    identifier = str(
        paper.get(
            "id",
            "unknown",
        )
    )

    try:
        return download_one(
            paper,
            root,
            policy,
        )

    except FetchError as exc:
        LOG.warning(
            "%s: acquisition failed; "
            "skipping paper (%s)",
            identifier,
            str(exc),
        )

        return {
            "id": identifier,
            "status": "unavailable",
            "failure_reason": (
                "fetch_error:"
                + type(exc).__name__
            ),
            "attempts": [
                {
                    "error": str(exc),
                }
            ],
        }

    except Exception as exc:
        # This is deliberately the final boundary around one paper.
        #
        # The exception is logged with a traceback so programming/data
        # problems are visible, but remaining papers still continue.
        LOG.exception(
            "%s: unexpected acquisition failure; "
            "skipping paper",
            identifier,
        )

        return {
            "id": identifier,
            "status": "unavailable",
            "failure_reason": (
                "worker_exception:"
                + type(exc).__name__
            ),
            "attempts": [
                {
                    "error":
                        type(exc).__name__,
                }
            ],
        }


# ============================================================
# CLI
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "data/raw/papers"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if not 1 <= args.workers <= 4:
        parser.error(
            "workers must be 1..4"
        )

    papers_path = (
        args.root
        / "papers.json"
    )

    if not papers_path.is_file():
        raise SystemExit(
            "Paper metadata file not found: "
            + str(papers_path)
        )

    papers = json.loads(
        papers_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        papers,
        list,
    ):
        raise SystemExit(
            "papers.json must contain "
            "a JSON array"
        )

    args.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(),

            logging.FileHandler(
                args.root
                / "download.log",
                encoding="utf-8",
            ),
        ],
    )

    LOG.info(
        "Starting acquisition for %s papers "
        "with %s workers",
        len(papers),
        args.workers,
    )

    policy = AccessPolicy()

    # --------------------------------------------------------
    # Important:
    #
    # Every worker calls download_one_safe(), not download_one().
    #
    # Therefore one paper raising IncompleteRead, timeout,
    # ConnectionResetError, malformed metadata, etc. cannot cause
    # executor.map() to abort all remaining papers.
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        results = list(
            executor.map(
                lambda paper: (
                    download_one_safe(
                        paper,
                        args.root,
                        policy,
                    )
                ),
                papers,
            )
        )

    downloaded_pdf = sum(
        result.get("format")
        == "pdf"
        for result in results
    )

    downloaded_html = sum(
        result.get("format")
        == "html"
        for result in results
    )

    unavailable = sum(
        result.get("status")
        == "unavailable"
        for result in results
    )

    cache_hits = sum(
        bool(
            result.get(
                "cache_hit"
            )
        )
        for result in results
    )

    worker_exceptions = sum(
        str(
            result.get(
                "failure_reason",
                "",
            )
        ).startswith(
            "worker_exception:"
        )
        for result in results
    )

    total_downloaded = (
        downloaded_pdf
        + downloaded_html
    )

    report = {
        "finished_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "selected_count":
            len(papers),

        "downloaded_pdf":
            downloaded_pdf,

        "downloaded_html":
            downloaded_html,

        "downloaded_total":
            total_downloaded,

        "unavailable":
            unavailable,

        "cache_hits":
            cache_hits,

        "worker_exceptions":
            worker_exceptions,

        # These are determined later by the extraction /
        # quality assessment pipeline.
        "fulltext_verified_count": 0,
        "training_ready_count": 0,

        "documents":
            results,
    }

    write_json(
        args.root
        / "download_report.json",
        report,
    )

    printable_report = {
        key: value
        for key, value
        in report.items()
        if key != "documents"
    }

    print(
        json.dumps(
            printable_report,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Batch policy
    #
    # Individual failed papers do NOT fail this command.
    #
    # If at least one source was acquired, extraction can proceed.
    # The later extractor enforces the real minimum-ready threshold.
    #
    # If zero papers were acquired, this is a systemic failure and
    # the workflow should stop.
    # --------------------------------------------------------

    if total_downloaded == 0:
        LOG.error(
            "No paper source documents were acquired."
        )

        return 1

    LOG.info(
        "Paper acquisition completed: "
        "%s downloaded, %s unavailable, "
        "%s worker exceptions",
        total_downloaded,
        unavailable,
        worker_exceptions,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )