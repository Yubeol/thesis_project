"""Collect traceable K-pop paper metadata; never invent missing paper sections.

When --existing is provided, papers already stored in PostgreSQL are filtered
before final selection. The collector then paginates deeper until it has enough
new unique candidates or reaches the configured page limit.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

CORE = re.compile(
    r"\bk[\s\-–]?pop\b|\bkorean (?:pop|wave)\b|\bhallyu\b|케이팝|한류",
    re.I,
)

CONTEXT = {
    "fandom": r"fan(?:dom|s|ning)?\b|팬덤|팬\s",
    "social_media": (
        r"social media|sns\b|tiktok|youtube|twitter|instagram|소셜|소셜미디어"
    ),
    "diffusion": (
        r"global|transnational|international|diffusion|circulation|해외|세계|글로벌"
    ),
    "participation": (
        r"participat|communit|activit|consum|팬활동|커뮤니티"
    ),
}

OPENALEX_WORK_ID = re.compile(
    r"(?:https?://)?openalex\.org/(W\d+)",
    re.I,
)


def write_json(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(path)


def normalize_title(
    title: str,
) -> str:
    return "".join(
        c
        for c in unicodedata.normalize(
            "NFKC",
            title,
        ).casefold()
        if c.isalnum()
    )


def normalize_doi(
    doi: str | None,
) -> str | None:
    value = unquote(
        doi or ""
    ).strip().lower()

    value = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        value,
    )

    return (
        value
        if re.fullmatch(
            r"10\.\d{4,9}/\S+",
            value,
        )
        else None
    )


def normalize_source_url(
    url: str | None,
) -> str | None:
    value = (
        url or ""
    ).strip()

    if not value:
        return None

    parsed = urlsplit(
        value
    )

    if (
        parsed.scheme not in {
            "http",
            "https",
        }
        or not parsed.hostname
    ):
        return None

    return value


def openalex_id_from_value(
    value: str | None,
) -> str | None:
    if not value:
        return None

    match = OPENALEX_WORK_ID.search(
        value.strip()
    )

    return (
        match.group(1).upper()
        if match
        else None
    )


def load_existing_index(
    path: Path | None,
) -> tuple[
    dict[str, set[str]],
    int,
]:
    """
    PostgreSQL snapshot을 읽어서
    OpenAlex 수집 전 중복 제거용 index를 만든다.
    """

    index = {
        "dois": set(),
        "source_urls": set(),
        "openalex_ids": set(),
    }

    if path is None:
        return index, 0

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    papers = payload.get(
        "papers"
    )

    if not isinstance(
        papers,
        list,
    ):
        raise ValueError(
            "Existing paper snapshot "
            "must contain a papers list"
        )

    for paper in papers:
        if not isinstance(
            paper,
            dict,
        ):
            continue

        doi = normalize_doi(
            paper.get("doi")
        )

        if doi:
            index[
                "dois"
            ].add(doi)

        source_url = (
            normalize_source_url(
                paper.get(
                    "source_url"
                )
            )
        )

        if source_url:
            index[
                "source_urls"
            ].add(source_url)

            work_id = (
                openalex_id_from_value(
                    source_url
                )
            )

            if work_id:
                index[
                    "openalex_ids"
                ].add(work_id)

        explicit_id = (
            openalex_id_from_value(
                paper.get(
                    "openalex_id"
                )
            )
        )

        if explicit_id:
            index[
                "openalex_ids"
            ].add(explicit_id)

    return (
        index,
        len(papers),
    )


def existing_match_reason(
    record: dict,
    index: dict[
        str,
        set[str],
    ],
) -> str | None:
    """
    이미 PostgreSQL에 있는 논문이면
    어떤 식별자로 매칭됐는지 반환한다.
    """

    doi = normalize_doi(
        record.get("doi")
    )

    if (
        doi
        and doi
        in index["dois"]
    ):
        return "doi"

    source_url = (
        normalize_source_url(
            record.get(
                "source_url"
            )
        )
    )

    if (
        source_url
        and source_url
        in index["source_urls"]
    ):
        return "source_url"

    work_id = str(
        record.get("id")
        or ""
    ).upper()

    if (
        work_id
        and work_id
        in index["openalex_ids"]
    ):
        return "openalex_id"

    return None


def abstract_text(
    index: dict | None,
) -> str | None:
    if not index:
        return None

    positions = {}

    for word, offsets in index.items():
        for position in offsets:
            positions[
                int(position)
            ] = word

    return (
        " ".join(
            positions[k]
            for k
            in sorted(positions)
        )
        or None
    )


def relevant(
    title: str,
    abstract: str | None,
) -> tuple[
    int,
    list[str],
]:
    combined = (
        title
        + " "
        + (abstract or "")
    )

    if not CORE.search(
        combined
    ):
        return 0, []

    matches = [
        name
        for name, pattern
        in CONTEXT.items()
        if re.search(
            pattern,
            combined,
            re.I,
        )
    ]

    if not matches:
        return 0, []

    score = (
        5
        * bool(
            CORE.search(
                title
            )
        )
        + len(matches)
    )

    score += sum(
        bool(
            re.search(
                pattern,
                title,
                re.I,
            )
        )
        for pattern
        in CONTEXT.values()
    )

    return (
        score,
        matches,
    )


def normalize_work(
    work: dict,
    query: str,
) -> dict | None:
    title = (
        work.get("title")
        or ""
    ).strip()

    abstract = abstract_text(
        work.get(
            "abstract_inverted_index"
        )
    )

    score, matches = relevant(
        title,
        abstract,
    )

    if (
        not title
        or not score
        or work.get(
            "is_retracted"
        )
    ):
        return None

    if work.get(
        "type"
    ) != "article":
        return None

    year = work.get(
        "publication_year"
    )

    if (
        not isinstance(
            year,
            int,
        )
        or year
        > datetime.now(
            timezone.utc
        ).year
    ):
        return None

    work_id = (
        work.get("id")
        or ""
    )

    if not re.fullmatch(
        r"https://openalex\.org/W\d+",
        work_id,
    ):
        return None

    doi = normalize_doi(
        work.get("doi")
    )

    locations = [
        work.get(
            "best_oa_location"
        )
        or {},
        *(
            work.get(
                "locations"
            )
            or []
        ),
    ]

    sources = []
    seen = set()

    for location in locations:
        if (
            not location
            or not location.get(
                "is_oa"
            )
        ):
            continue

        for kind in (
            "pdf_url",
            "landing_page_url",
        ):
            url = location.get(
                kind
            )

            if (
                not url
                or url in seen
            ):
                continue

            if (
                urlsplit(
                    url
                ).scheme
                not in {
                    "http",
                    "https",
                }
            ):
                continue

            seen.add(url)

            sources.append(
                {
                    "url": url,
                    "kind": kind,
                    "license": (
                        location.get(
                            "license"
                        )
                    ),
                    "version": (
                        location.get(
                            "version"
                        )
                    ),
                }
            )

    authors = [
        (
            authorship.get(
                "author"
            )
            or {}
        ).get(
            "display_name"
        )
        for authorship
        in (
            work.get(
                "authorships"
            )
            or []
        )
    ]

    authors = [
        author
        for author in authors
        if author
    ]

    keywords = [
        keyword.get(
            "display_name"
        )
        for keyword
        in (
            work.get(
                "keywords"
            )
            or []
        )
        if keyword.get(
            "display_name"
        )
    ]

    return {
        "id": (
            work_id.rsplit(
                "/",
                1,
            )[-1]
        ),

        "title":
            title,

        "authors":
            authors,

        "published_year":
            year,

        "abstract":
            abstract,

        "introduction":
            None,

        "body":
            None,

        "conclusion":
            None,

        "keywords":
            keywords,

        "source":
            "OpenAlex",

        "source_url": (
            "https://doi.org/"
            + doi
            if doi
            else work_id
        ),

        "metadata_url":
            work_id,

        "doi":
            doi,

        "language":
            work.get(
                "language"
            ),

        "is_oa":
            bool(
                (
                    work.get(
                        "open_access"
                    )
                    or {}
                ).get(
                    "is_oa"
                )
            ),

        "oa_status":
            (
                work.get(
                    "open_access"
                )
                or {}
            ).get(
                "oa_status"
            ),

        "source_locations":
            sources,

        "matched_topics":
            matches,

        "relevance_score":
            score,

        "matched_queries": [
            query
        ],

        "fulltext_status":
            "not_acquired",

        "section_status":
            "not_extracted",
    }


def select_unique(
    records: list[dict],
    target: int,
) -> list[dict]:
    ranked = sorted(
        records,
        key=lambda record: (
            -int(
                record[
                    "is_oa"
                ]
            ),
            -record[
                "relevance_score"
            ],
            record["id"],
        ),
    )

    result = []
    identifiers = {}

    for record in ranked:
        keys = [
            (
                "id",
                record["id"],
            ),
            (
                "title",
                normalize_title(
                    record["title"]
                ),
            ),
        ]

        if record["doi"]:
            keys.append(
                (
                    "doi",
                    record["doi"],
                )
            )

        previous = next(
            (
                identifiers[key]
                for key in keys
                if key
                in identifiers
            ),
            None,
        )

        if previous is not None:
            existing = result[
                previous
            ]

            existing[
                "matched_queries"
            ] = sorted(
                set(
                    existing[
                        "matched_queries"
                    ]
                    + record[
                        "matched_queries"
                    ]
                )
            )

            known_urls = {
                location["url"]
                for location
                in existing[
                    "source_locations"
                ]
            }

            existing[
                "source_locations"
            ] += [
                location
                for location
                in record[
                    "source_locations"
                ]
                if location[
                    "url"
                ]
                not in known_urls
            ]

            if (
                record["id"]
                != existing["id"]
            ):
                existing[
                    "duplicate_metadata_ids"
                ] = sorted(
                    set(
                        existing.get(
                            "duplicate_metadata_ids",
                            [],
                        )
                        + [
                            record[
                                "id"
                            ]
                        ]
                    )
                )

            for key in keys:
                identifiers[
                    key
                ] = previous

            continue

        for key in keys:
            identifiers[
                key
            ] = len(result)

        result.append(
            {
                **record,
                "source_locations":
                    list(
                        record[
                            "source_locations"
                        ]
                    ),
            }
        )

    return result[:target]


def fetch_openalex_page(
    *,
    query: str,
    cursor: str,
    config: dict,
    output: Path,
    api_key: str,
    refresh: bool,
) -> tuple[
    dict,
    Path,
    bool,
]:
    base_params = {
        "search": query,

        "filter": (
            "type:article,"
            "is_retracted:false"
        ),

        "per_page":
            config[
                "page_size"
            ],

        "cursor":
            cursor,
    }

    if config[
        "open_access_only"
    ]:
        base_params[
            "filter"
        ] += ",is_oa:true"

    cache_id = (
        hashlib.sha256(
            json.dumps(
                base_params,
                sort_keys=True,
            ).encode()
        )
        .hexdigest()[:24]
    )

    cache = (
        output
        / "api"
        / (
            cache_id
            + ".json"
        )
    )

    if (
        cache.exists()
        and not refresh
    ):
        response = json.loads(
            cache.read_text(
                encoding="utf-8"
            )
        )

        fetched = False

    else:
        request_params = dict(
            base_params
        )

        if api_key:
            request_params[
                "api_key"
            ] = api_key

        response = get_json(
            "https://api.openalex.org/works",
            request_params,
            headers={},
        )

        if not isinstance(
            response.get(
                "results"
            ),
            list,
        ):
            raise FetchError(
                "OpenAlex response "
                "is missing results"
            )

        write_json(
            cache,
            response,
        )

        fetched = True

        time.sleep(1)

    if not isinstance(
        response.get(
            "results"
        ),
        list,
    ):
        raise FetchError(
            "OpenAlex response "
            "is missing results"
        )

    return (
        response,
        cache,
        fetched,
    )


def collect(
    config: dict,
    output: Path,
    *,
    refresh: bool = False,
    existing: Path | None = None,
) -> dict:
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    errors = []

    raw_count = 0
    cached_pages = 0
    fetched_pages = 0

    relevant_before_existing_filter = 0
    existing_filtered = 0

    match_reasons: Counter[
        str
    ] = Counter()

    (
        existing_index,
        existing_snapshot_count,
    ) = load_existing_index(
        existing
    )

    LOG.info(
        "Existing snapshot: "
        "papers=%s "
        "doi=%s "
        "source_url=%s "
        "openalex_id=%s",

        existing_snapshot_count,
        len(
            existing_index[
                "dois"
            ]
        ),
        len(
            existing_index[
                "source_urls"
            ]
        ),
        len(
            existing_index[
                "openalex_ids"
            ]
        ),
    )

    api_key = (
        os.getenv(
            "OPENALEX_API_KEY",
            "",
        )
        .strip()
    )

    if api_key:
        LOG.info(
            "OpenAlex API key configured."
        )

    else:
        LOG.warning(
            "OPENALEX_API_KEY "
            "is not configured."
        )

    states = [
        {
            "query":
                query,

            "cursor":
                "*",

            "active":
                True,

            "pages":
                0,
        }
        for query
        in config["queries"]
    ]

    target = config[
        "target"
    ]

    page_limit = config[
        "pages_per_query"
    ]

    # Query 1의 결과만 잔뜩 뽑히지 않도록
    # query별로 1페이지씩 round-robin 방식으로 진행한다.
    for round_number in range(
        1,
        page_limit + 1,
    ):
        fetched_any = False

        for state in states:
            if not state[
                "active"
            ]:
                continue

            query = state[
                "query"
            ]

            cursor = state[
                "cursor"
            ]

            page_number = (
                state["pages"]
                + 1
            )

            try:
                (
                    response,
                    cache,
                    fetched,
                ) = fetch_openalex_page(
                    query=query,
                    cursor=cursor,
                    config=config,
                    output=output,
                    api_key=api_key,
                    refresh=refresh,
                )

                fetched_any = True

                state[
                    "pages"
                ] = page_number

                fetched_pages += int(
                    fetched
                )

                cached_pages += int(
                    not fetched
                )

                works = response[
                    "results"
                ]

                raw_count += len(
                    works
                )

                for work in works:
                    record = (
                        normalize_work(
                            work,
                            query,
                        )
                    )

                    if not record:
                        continue

                    if (
                        config[
                            "open_access_only"
                        ]
                        and not record[
                            "is_oa"
                        ]
                    ):
                        continue

                    relevant_before_existing_filter += 1

                    reason = (
                        existing_match_reason(
                            record,
                            existing_index,
                        )
                    )

                    if reason:
                        existing_filtered += 1

                        match_reasons[
                            reason
                        ] += 1

                        continue

                    record[
                        "raw_metadata_path"
                    ] = (
                        cache.relative_to(
                            output
                        )
                        .as_posix()
                    )

                    records.append(
                        record
                    )

                LOG.info(
                    "Query %r page %s: "
                    "raw=%s accumulated_new=%s",

                    query,
                    page_number,
                    len(works),
                    len(records),
                )

                next_cursor = (
                    response.get(
                        "meta"
                    )
                    or {}
                ).get(
                    "next_cursor"
                )

                if (
                    len(works)
                    < config[
                        "page_size"
                    ]
                    or not next_cursor
                    or next_cursor
                    == cursor
                ):
                    state[
                        "active"
                    ] = False

                else:
                    state[
                        "cursor"
                    ] = next_cursor

            except (
                FetchError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                errors.append(
                    {
                        "query":
                            query,

                        "page":
                            page_number,

                        "error":
                            str(exc),
                    }
                )

                LOG.error(
                    "Query failed "
                    "%r page %s: %s",

                    query,
                    page_number,
                    exc,
                )

                state[
                    "active"
                ] = False

        unique_candidates = (
            select_unique(
                records,
                len(records),
            )
        )

        LOG.info(
            "Round %s complete: "
            "unique new candidates=%s "
            "target=%s",

            round_number,
            len(
                unique_candidates
            ),
            target,
        )

        # 모든 query가 같은 depth까지 간 다음
        # 신규 목표 개수를 채웠으면 종료한다.
        if (
            len(
                unique_candidates
            )
            >= target
        ):
            break

        if (
            not fetched_any
            or not any(
                state["active"]
                for state
                in states
            )
        ):
            break

    unique_candidates = (
        select_unique(
            records,
            len(records),
        )
    )

    selected = (
        unique_candidates[
            :target
        ]
    )

    report = {
        "schema_version":
            2,

        "collected_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "config":
            config,

        "existing_snapshot_count":
            existing_snapshot_count,

        "raw_results":
            raw_count,

        "relevant_results_before_existing_filter":
            relevant_before_existing_filter,

        "existing_filtered":
            existing_filtered,

        "existing_match_reasons":
            dict(
                sorted(
                    match_reasons.items()
                )
            ),

        "relevant_results_before_dedup":
            len(records),

        "unique_new_candidates":
            len(
                unique_candidates
            ),

        # 기존 로그/테스트와의 호환용
        "unique_candidates":
            len(
                unique_candidates
            ),

        "selected_count":
            len(selected),

        "new_target":
            target,

        "new_target_met":
            len(selected)
            >= target,

        "with_abstract":
            sum(
                bool(
                    record[
                        "abstract"
                    ]
                )
                for record
                in selected
            ),

        "with_source_location":
            sum(
                bool(
                    record[
                        "source_locations"
                    ]
                )
                for record
                in selected
            ),

        "cached_pages":
            cached_pages,

        "fetched_pages":
            fetched_pages,

        "pages_by_query": {
            state["query"]:
                state["pages"]
            for state
            in states
        },

        "metadata_minimum_met":
            len(selected)
            >= config[
                "minimum"
            ],

        "fulltext_verified_count":
            0,

        "training_ready_count":
            0,

        "errors":
            errors,
    }

    write_json(
        output
        / "papers.json",
        selected,
    )

    write_json(
        output
        / "collection_report.json",
        report,
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/raw/papers"
        ),
    )

    parser.add_argument(
        "--target",
        type=int,
    )

    parser.add_argument(
        "--minimum",
        type=int,
    )

    parser.add_argument(
        "--existing",
        type=Path,
        help=(
            "JSON snapshot of papers "
            "already stored in PostgreSQL"
        ),
    )

    parser.add_argument(
        "--pages-per-query",
        type=int,
        help=(
            "Override maximum "
            "OpenAlex cursor pages "
            "searched per query"
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
    )

    args = parser.parse_args()

    config = json.loads(
        args.config.read_text(
            encoding="utf-8"
        )
    )

    for name in (
        "target",
        "minimum",
        "pages_per_query",
    ):
        value = getattr(
            args,
            name,
        )

        if value is not None:
            config[
                name
            ] = value

    if not (
        1
        <= config[
            "minimum"
        ]
        <= config[
            "target"
        ]
        <= 1000
    ):
        parser.error(
            "Require 1 <= minimum "
            "<= target <= 1000"
        )

    if not (
        1
        <= config[
            "page_size"
        ]
        <= 200
    ):
        parser.error(
            "page_size "
            "must be 1..200"
        )

    if not (
        1
        <= config[
            "pages_per_query"
        ]
        <= 20
    ):
        parser.error(
            "pages_per_query "
            "must be 1..20"
        )

    if (
        not config[
            "queries"
        ]
        or not all(
            isinstance(
                query,
                str,
            )
            and query.strip()
            for query
            in config[
                "queries"
            ]
        )
    ):
        parser.error(
            "Provide non-empty "
            "search queries"
        )

    if (
        args.existing
        is not None
        and not args.existing.is_file()
    ):
        parser.error(
            "--existing snapshot "
            "file does not exist"
        )

    args.output.mkdir(
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
                args.output
                / "collection.log",
                encoding="utf-8",
            ),
        ],
    )

    try:
        report = collect(
            config,
            args.output,
            refresh=args.refresh,
            existing=args.existing,
        )

    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        LOG.error(
            "Collector "
            "configuration/state "
            "failure: %s",
            type(exc).__name__,
        )

        return 1

    print(
        json.dumps(
            report,
            ensure_ascii=True,
            indent=2,
        )
    )

    return (
        0
        if (
            report[
                "metadata_minimum_met"
            ]
            and not report[
                "errors"
            ]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )