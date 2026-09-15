"""Bounded, retrying HTTP reads. Never log request headers or query secrets."""

from __future__ import annotations

import json
import logging
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

LOG = logging.getLogger(__name__)
USER_AGENT = "PaperAIAgent/0.1 (academic corpus collection)"


class FetchError(RuntimeError):
    pass


def retry_delay(value: str | None, attempt: int) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            seconds = 2 ** attempt
    return min(30.0, max(0.0, seconds))


def fetch(url: str, *, headers: dict | None = None, attempts: int = 3,
          timeout: float = 20, max_bytes: int = 25_000_000) -> tuple[bytes, str, str]:
    host = urlsplit(url).hostname
    if urlsplit(url).scheme not in {"http", "https"} or not host:
        raise FetchError("Only HTTP(S) source URLs are accepted")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise FetchError(f"Response exceeds byte limit from {host}")
                return payload, response.headers.get_content_type(), response.url
        except HTTPError as exc:
            exc.close()
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise FetchError(f"HTTP {exc.code} from {host}") from None
            delay = retry_delay(exc.headers.get("Retry-After"), attempt)
            LOG.warning("Retry HTTP %s from %s (%s/%s)", exc.code, host, attempt + 1, attempts)
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == attempts - 1:
                raise FetchError(f"{type(exc).__name__} from {host}") from None
            delay = retry_delay(None, attempt)
            LOG.warning("Retry network failure from %s (%s/%s)", host, attempt + 1, attempts)
        time.sleep(delay)
    raise AssertionError("unreachable")


def get_json(base: str, params: dict, *, headers: dict | None = None) -> dict:
    payload, _, _ = fetch(base + "?" + urlencode(params), headers=headers)
    try:
        result = json.loads(payload)
    except (ValueError, UnicodeError):
        raise FetchError(f"Invalid JSON from {urlsplit(base).hostname}") from None
    if not isinstance(result, dict):
        raise FetchError("Expected JSON object")
    return result
