from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from transformer.preprocessing.prompts import (
    HEADINGS,
    clean_text,
    make_input,
)


FIELDS = (
    "paper_id",
    "title",
    "keywords",
    "abstract",
    "introduction",
    "body",
    "conclusion",
    "language",
    "doi",
    "content_hash",
)

SPLITS = (
    "train",
    "validation",
    "test",
)

SECTION_FIELDS = {
    "Introduction": "introduction",
    "Body": "body",
    "Conclusion": "conclusion",
}

# 각 section의 학습 Target 길이.
# 너무 짧은 한 문장짜리 Target을 방지한다.
TARGET_WORD_LIMITS = {
    "Introduction": (50, 150),
    "Body": (100, 260),
    "Conclusion": (40, 130),
}

# 논문 본문에 섞여 들어오는 메타데이터/잡음 제거용
METADATA_PATTERNS = (
    r"^\s*(?:abstract|keywords?|references?|bibliography)\s*:?\s*$",
    r"^\s*(?:received|accepted|published|copyright|corresponding author)\b",
    r"^\s*(?:vol(?:ume)?\.?|issue|pages?|pp\.?|issn)\s*[:\d]",
    r"https?://",
    r"\bdoi\s*:",
    r"©",
    r"all rights reserved",
)


def export_postgres(env_file=None):
    """
    EC2 PostgreSQL의 public.papers를 read-only로 가져온다.
    """
    from pipeline.common.database import DEFAULT_ENV, connect

    with connect(
        Path(env_file) if env_file else DEFAULT_ENV,
        read_only=True,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'papers'
                """
            )

            schema = dict(cursor.fetchall())

            missing = set(FIELDS) - set(schema)

            if missing:
                raise ValueError(
                    "public.papers is missing columns: "
                    + ", ".join(sorted(missing))
                )

            cursor.execute(
                """
                SELECT
                    paper_id,
                    title,
                    keywords,
                    abstract,
                    introduction,
                    body,
                    conclusion,
                    language,
                    doi,
                    content_hash
                FROM public.papers
                ORDER BY paper_id
                """
            )

            rows = [
                dict(zip(FIELDS, row))
                for row in cursor.fetchall()
            ]

            return rows, schema


def normalized(value):
    """
    중복 검사 및 문자열 비교용 정규화.
    """
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        unicodedata.normalize(
            "NFKC",
            clean_text(value),
        ).lower(),
    ).strip()


def is_metadata_sentence(text):
    text = clean_text(text)

    if not text:
        return True

    return any(
        re.search(pattern, text, re.I)
        for pattern in METADATA_PATTERNS
    )


def sentences(text):
    """
    논문 원문에서 학습에 사용할 수 있는 영어 문장만 추출한다.
    """
    text = clean_text(text)

    if not text:
        return []

    # [1], [1, 2], [3-5] 같은 citation marker 제거
    text = re.sub(
        r"\[(?:\d+[\s,;–-]*)+\]",
        "",
        text,
    )

    # (Smith, 2020) 형태의 단순 citation 제거
    text = re.sub(
        r"\([^)]*\b(?:19|20)\d{2}[a-z]?[^)]*\)",
        "",
        text,
    )

    candidates = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    result = []

    for sentence in candidates:
        sentence = clean_text(sentence)

        if not sentence:
            continue

        if is_metadata_sentence(sentence):
            continue

        words = sentence.split()

        # 너무 짧거나 비정상적으로 긴 문장 제외
        if not 8 <= len(words) <= 90:
            continue

        english_ratio = (
            len(re.findall(r"[a-zA-Z]", sentence))
            / max(1, len(sentence))
        )

        if english_ratio <= 0.55:
            continue

        result.append(sentence)

    return result


def build_section_target(text, section):
    """
    실제 Introduction / Body / Conclusion에서
    여러 문장을 모아 section 단위 TARGET을 만든다.

    예전처럼 문장 하나만 Target으로 사용하지 않는다.
    """
    minimum_words, maximum_words = TARGET_WORD_LIMITS[section]

    chosen = []
    word_count = 0

    for sentence in sentences(text):
        sentence_words = len(sentence.split())

        if (
            word_count >= minimum_words
            and word_count + sentence_words > maximum_words
        ):
            break

        chosen.append(sentence)
        word_count += sentence_words

        if word_count >= maximum_words:
            break

    if word_count < minimum_words:
        raise ValueError(
            f"section_too_short_{section.lower()}"
        )

    return " ".join(chosen)


def select_evidence(
    row,
    target_section,
    target_text,
    keywords,
    max_items=4,
):
    """
    Target과 동일한 문장을 Evidence에 넣지 않는다.

    Evidence 우선순위:
    1. abstract
    2. 현재 Target이 아닌 다른 section

    예:
    Introduction을 생성하는 샘플이라면
    Evidence는 abstract + body/conclusion에서 가져올 수 있다.
    """
    target_sentence_norms = {
        normalized(sentence)
        for sentence in sentences(target_text)
    }

    query_terms = set(
        normalized(
            " ".join(
                [
                    clean_text(row.get("title")),
                    *keywords[:8],
                ]
            )
        ).split()
    )

    query_terms -= {
        "the",
        "and",
        "of",
        "in",
        "to",
        "a",
        "an",
        "for",
        "on",
        "with",
    }

    # abstract는 evidence로 가장 우선 사용
    sources = [
        (
            "abstract",
            row.get("abstract"),
            3,
        ),
    ]

    # 현재 Target section이 아닌 다른 section도 Evidence 후보로 사용
    for section, field in SECTION_FIELDS.items():
        if section != target_section:
            sources.append(
                (
                    field,
                    row.get(field),
                    1,
                )
            )

    ranked = []
    seen = set()
    order = 0

    for _, text, source_bonus in sources:
        for sentence in sentences(text):
            norm = normalized(sentence)

            if not norm:
                continue

            # Target과 동일한 문장은 Evidence에서 제외
            if norm in target_sentence_norms:
                continue

            if norm in seen:
                continue

            seen.add(norm)

            overlap = len(
                query_terms
                & set(norm.split())
            )

            score = source_bonus + overlap

            ranked.append(
                (
                    score,
                    -order,
                    sentence,
                )
            )

            order += 1

    ranked.sort(reverse=True)

    evidence = [
        sentence
        for _, _, sentence in ranked[:max_items]
    ]

    if not evidence:
        raise ValueError(
            f"missing_evidence_{target_section.lower()}"
        )

    return evidence


def identity_keys(row):
    """
    동일 논문 또는 사실상 동일한 논문이
    서로 다른 train/validation/test split으로 갈라지는 것을 방지한다.
    """
    keys = [
        (
            "id",
            str(row["paper_id"]),
        )
    ]

    if row.get("doi"):
        doi = re.sub(
            r"^(https?://(dx\.)?doi.org/|doi:\s*)",
            "",
            row["doi"].lower(),
        ).strip()

        keys.append(
            (
                "doi",
                doi,
            )
        )

    if row.get("content_hash"):
        keys.append(
            (
                "hash",
                row["content_hash"],
            )
        )

    title = " ".join(
        token
        for token in normalized(row["title"]).split()
        if token
        not in {
            "a",
            "an",
            "the",
            "in",
            "on",
            "of",
            "and",
            "to",
        }
    )

    keys.append(
        (
            "title",
            title,
        )
    )

    content = "\n".join(
        normalized(row.get(key))
        for key in (
            "introduction",
            "body",
            "conclusion",
        )
    )

    keys.append(
        (
            "text",
            hashlib.sha256(
                content.encode()
            ).hexdigest(),
        )
    )

    return keys


def prepare_paper(row):
    """
    논문 한 편을 section 학습용으로 준비한다.
    """
    if (
        not isinstance(
            row.get("paper_id"),
            (str, int),
        )
        or isinstance(
            row.get("paper_id"),
            bool,
        )
    ):
        raise ValueError(
            "missing_paper_id"
        )

    if row.get("language") not in (
        "en",
        "eng",
        "english",
    ):
        raise ValueError(
            "non_english_or_unknown_language"
        )

    if not clean_text(row.get("title")):
        raise ValueError(
            "missing_title"
        )

    keywords = row.get("keywords") or []

    if (
        not isinstance(keywords, list)
        or any(
            not isinstance(item, str)
            for item in keywords
        )
    ):
        raise ValueError(
            "invalid_keywords"
        )

    targets = {}
    evidence_by_section = {}

    for section in HEADINGS:
        field = SECTION_FIELDS[section]

        target = build_section_target(
            row.get(field),
            section,
        )

        evidence = select_evidence(
            row=row,
            target_section=section,
            target_text=target,
            keywords=keywords,
        )

        # 마지막 안전 검사:
        # Target과 Evidence가 완전히 동일한 문장을 공유하는지 확인
        target_norms = {
            normalized(sentence)
            for sentence in sentences(target)
        }

        evidence_norms = {
            normalized(sentence)
            for item in evidence
            for sentence in sentences(item)
        }

        if target_norms & evidence_norms:
            raise ValueError(
                "target_evidence_overlap_"
                + section.lower()
            )

        targets[section] = target
        evidence_by_section[section] = evidence

    return {
        "row": row,
        "targets": targets,
        "evidence": evidence_by_section,
    }


def build_samples(
    rows,
    seed=42,
):
    """
    논문 단위로 split한 뒤,
    한 논문에서 Introduction / Body / Conclusion
    총 3개의 학습 sample을 생성한다.
    """
    rejected = Counter()
    eligible = []

    for row in rows:
        try:
            eligible.append(
                prepare_paper(row)
            )
        except ValueError as exc:
            rejected[str(exc)] += 1

    # ---------------------------------------------------------
    # duplicate paper를 split 전에 하나의 group으로 묶는다.
    # ---------------------------------------------------------
    parent = list(
        range(len(eligible))
    )

    def root(index):
        while parent[index] != index:
            parent[index] = parent[
                parent[index]
            ]

            index = parent[index]

        return index

    seen = {}

    for index, entry in enumerate(eligible):
        row = entry["row"]

        for key in identity_keys(row):
            if key in seen:
                parent[root(index)] = root(
                    seen[key]
                )

            seen[key] = index

    groups = {}

    for index, entry in enumerate(eligible):
        groups.setdefault(
            root(index),
            [],
        ).append(entry)

    groups = sorted(
        groups.values(),
        key=lambda group: min(
            str(
                item["row"]["paper_id"]
            )
            for item in group
        ),
    )

    random.Random(seed).shuffle(groups)

    if len(groups) < 3:
        raise ValueError(
            "At least three eligible, distinct English papers "
            "are required for train/validation/test"
        )

    n_validation = max(
        1,
        round(
            len(groups) * 0.1
        ),
    )

    n_test = max(
        1,
        round(
            len(groups) * 0.1
        ),
    )

    n_train = (
        len(groups)
        - n_validation
        - n_test
    )

    assignments = (
        ["train"] * n_train
        + ["validation"] * n_validation
        + ["test"] * n_test
    )

    output = {
        name: []
        for name in SPLITS
    }

    paper_counts = Counter()

    for group, split in zip(
        groups,
        assignments,
    ):
        # duplicate group 중 대표 논문 하나 선택
        representative = min(
            group,
            key=lambda item: str(
                item["row"]["paper_id"]
            ),
        )

        row = representative["row"]

        keywords = row.get("keywords") or []

        source_paper_ids = sorted(
            {
                str(
                    item["row"]["paper_id"]
                )
                for item in group
            }
        )

        # -----------------------------------------------------
        # 핵심:
        # 논문 한 편 → 3개의 section sample
        # -----------------------------------------------------
        for section in HEADINGS:
            value = make_input(
                title=row["title"],
                topic=(
                    ", ".join(keywords[:4])
                    or row["title"]
                ),
                paper_evidence=(
                    representative[
                        "evidence"
                    ][section]
                ),
                news_evidence=[],
                section=section,
            )

            output[split].append(
                {
                    "paper_id": row["paper_id"],

                    "source_paper_ids": (
                        source_paper_ids
                    ),

                    "section": section,

                    "input": value,

                    "target": (
                        representative[
                            "targets"
                        ][section]
                    ),

                    "target_kind": (
                        "sectional_weak_supervision_v2"
                    ),
                }
            )

        paper_counts[split] += 1

    report = {
        "source_papers": len(rows),

        "eligible_papers": (
            len(eligible)
        ),

        "unique_paper_groups": (
            len(groups)
        ),

        "duplicates_grouped": (
            len(eligible)
            - len(groups)
        ),

        "rejected": dict(
            rejected
        ),

        # 실제 논문 수
        "paper_counts": {
            name: paper_counts[name]
            for name in SPLITS
        },

        # 실제 학습 sample 수
        # 정상이라면 paper_counts × 3
        "counts": {
            name: len(
                output[name]
            )
            for name in SPLITS
        },
    }

    return output, report


def write_dataset(
    samples,
    report,
    output_dir,
    seed,
    schema=None,
):
    """
    train / validation / test JSONL 및 manifest 저장.
    기존 snapshot을 덮어쓰지 않는다.
    """
    output_dir = Path(
        output_dir
    )

    targets = [
        output_dir / f"{name}.jsonl"
        for name in SPLITS
    ]

    targets.append(
        output_dir
        / "manifest.json"
    )

    if any(
        path.exists()
        for path in targets
    ):
        raise ValueError(
            "Dataset output already exists; "
            "choose a new --output-dir to preserve the previous snapshot"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hashes = {}

    for split in SPLITS:
        content = "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in samples[split]
        ).encode(
            "utf-8"
        )

        path = (
            output_dir
            / f"{split}.jsonl"
        )

        temporary = (
            path.with_suffix(
                ".tmp"
            )
        )

        temporary.write_bytes(
            content
        )

        temporary.replace(
            path
        )

        hashes[split] = (
            hashlib.sha256(
                content
            ).hexdigest()
        )

    manifest = {
        "schema_version": 2,

        "seed": seed,

        "split_ratio": [
            0.8,
            0.1,
            0.1,
        ],

        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "sha256": hashes,

        "source_schema": (
            schema
            or {}
        ),

        **report,

        "limitations": (
            "Section-level weak supervision from source papers. "
            "Targets are no longer direct copies of the evidence. "
            "News is not yet paired and uses NO_NEWS_EVIDENCE. "
            "Targets are not human-reviewed abstractive drafts. "
            "Near-duplicate semantic papers may still require review."
        ),
    }

    (
        output_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build section-level Transformer training data "
            "from PostgreSQL papers"
        )
    )

    parser.add_argument(
        "--env-file",
        type=Path,
    )

    parser.add_argument(
        "--papers-json",
        type=Path,
        help=(
            "Optional offline PostgreSQL-shaped JSON list "
            "containing real paper_id values"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/training"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    try:
        if args.papers_json:
            rows = json.loads(
                args.papers_json.read_text(
                    encoding="utf-8-sig"
                )
            )

            schema = None

            if not isinstance(
                rows,
                list,
            ):
                raise ValueError(
                    "--papers-json must contain a JSON list"
                )

        else:
            rows, schema = export_postgres(
                args.env_file
            )

        samples, report = build_samples(
            rows,
            args.seed,
        )

        manifest = write_dataset(
            samples,
            report,
            args.output_dir,
            args.seed,
            schema,
        )

        print(
            json.dumps(
                {
                    "output_dir": str(
                        args.output_dir
                    ),

                    "schema_version": (
                        manifest[
                            "schema_version"
                        ]
                    ),

                    **report,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        return 0

    except Exception as exc:
        # DB exception에 접속정보가 포함될 수 있으므로
        # 직접 만든 ValueError만 상세 출력한다.
        message = (
            str(exc)
            if isinstance(
                exc,
                ValueError,
            )
            else type(exc).__name__
        )

        parser.exit(
            1,
            "Dataset export failed: "
            + message
            + "\n",
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
