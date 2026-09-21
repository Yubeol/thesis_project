from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from transformer.preprocessing.prompts import (
    HEADINGS,
    clean_text,
    encode_input,
    encode_target,
    format_evidence_item,
    make_input,
    render_prompt,
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
    "Introduction": (50, 130),
    "Body": (100, 180),
    "Conclusion": (40, 110),
}

SUPPORT_STOPWORDS = {
    "about", "across", "after", "also", "among", "based", "been", "between",
    "could", "from", "have", "into", "more", "most", "other", "over",
    "paper", "research", "result", "results", "show", "shows", "study",
    "such", "than", "that", "their", "there", "these", "this", "those",
    "through", "under", "using", "were", "what", "when", "where", "which",
    "while", "with", "would", "kpop", "korean", "culture", "social", "media",
    "global", "music", "fandom", "fans", "digital", "online", "analysis",
    "the", "and", "for", "are", "not", "its", "can", "has", "was", "one",
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
    r"\b(?:table|figure|appendix|contents)\s+\d+\b",
    r"\b(?:references|bibliography)\b",
    r"\b(?:university|department|faculty)\s+of\b.*\bemail\b",
    r"\b(?:bachelor.?s student|department of|faculty of|university of)\b",
    r"\b(?:CFI|TLI|RMSEA|SRMR)\b",
)


def export_news_postgres(env_file=None):
    """Read only the existing English RAG-ready news; never pair by recency alone."""
    from pipeline.common.database import DEFAULT_ENV, connect

    with connect(Path(env_file) if env_file else DEFAULT_ENV, read_only=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT news_id, title_en_for_rag, content_en_for_rag, source
                   FROM public.news
                   WHERE title_en_for_rag IS NOT NULL
                     AND content_en_for_rag IS NOT NULL
                   ORDER BY news_id"""
            )
            return [
                {"news_id": news_id, "title": title, "content": content, "source": source}
                for news_id, title, content, source in cursor.fetchall()
            ]


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


def content_terms(value):
    return {
        token for token in normalized(value).split()
        if len(token) >= 4 and token not in SUPPORT_STOPWORDS
    }


def is_near_copy(left, right):
    a, b = normalized(left), normalized(right)
    if a == b:
        return True
    left_terms, right_terms = set(a.split()), set(b.split())
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms) / len(left_terms | right_terms)
    if overlap >= 0.82:
        return True
    # Expensive character comparison is only useful for plausible copies.
    return overlap >= 0.55 and SequenceMatcher(None, a, b).ratio() >= 0.84


def support_score(target, evidence):
    target_terms = content_terms(target)
    evidence_terms = content_terms(" ".join(evidence))
    return len(target_terms & evidence_terms) / max(1, len(target_terms))


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
        if not 8 <= len(words) <= 70:
            continue

        if not re.search(r"[.!?][\"'”’]?\s*$", sentence):
            continue

        if re.search(r"(?:\.{3,}|[_=]{3,}|\s[,;:]\s[,;:])", sentence):
            continue

        numeric_words = sum(bool(re.search(r"\d", word)) for word in words)
        if numeric_words >= 4 and numeric_words / len(words) >= 0.12:
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
    """Rank non-copy sentences by their support for the requested target."""
    target_sentences = sentences(target_text)
    target_terms = content_terms(target_text)
    sources = [("abstract", row.get("abstract"), 2)]
    sources.extend(
        (field, row.get(field), 1)
        for section, field in SECTION_FIELDS.items()
        if section != target_section
    )
    # Later sentences in the same section may support an earlier target.
    sources.append((target_section.lower(), row.get(SECTION_FIELDS[target_section]), 0))
    ranked, seen = [], set()

    for source_name, text, bonus in sources:
        for order, sentence in enumerate(sentences(text)):
            norm = normalized(sentence)
            if norm in seen:
                continue
            terms = content_terms(sentence)
            overlap = len(terms & target_terms)
            if overlap < 2:
                continue
            if any(is_near_copy(sentence, target) for target in target_sentences):
                continue
            seen.add(norm)
            score = overlap / max(1, len(terms)) + overlap / max(1, len(target_terms))
            ranked.append((score, bonus, -order, sentence, source_name))

    ranked.sort(reverse=True)
    selected = ranked[:max_items]
    if not selected:
        raise ValueError(f"missing_evidence_{target_section.lower()}")
    return [(sentence, source_name) for _, _, _, sentence, source_name in selected]


def prepare_news_rows(news_rows):
    prepared = []
    for news in news_rows or []:
        title = clean_text(news.get("title"))
        source = clean_text(news.get("source"))
        candidate_sentences = sentences(news.get("content"))[:8]
        if not title or not source or not candidate_sentences:
            continue
        title_terms = content_terms(title)
        prepared.append({
            "news_id": str(news["news_id"]),
            "title": title,
            "source": source,
            "sentences": candidate_sentences,
            "title_terms": title_terms,
            "article_terms": content_terms(" ".join(candidate_sentences)) | title_terms,
            "sentence_terms": [content_terms(sentence) for sentence in candidate_sentences],
        })
    return prepared


def named_title_anchors(title):
    """Extract named entities, excluding generic Korean-pop words."""
    return {
        token.casefold()
        for token in re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z][A-Za-z]*)\b", title or "")
        if token.casefold() not in {"kpop", "korean", "hallyu", "ip", "pop"}
    }


def select_news_evidence(row, target, news_rows, max_items=1):
    """Require a paper-specific named subject in both target and news."""
    anchors = content_terms(row["title"])
    platforms = {"tiktok", "youtube", "instagram", "facebook", "twitter", "netflix"}
    named_anchors = named_title_anchors(row["title"]) - platforms
    target_terms = content_terms(target)
    target_named = named_anchors & target_terms
    ranked = []
    for news in news_rows or []:
        title = news["title"]
        source = news["source"]
        candidate_sentences = news["sentences"]
        title_terms = news["title_terms"]
        article_terms = news["article_terms"]
        if not target_named or not (target_named & article_terms):
            continue
        shared_subject_in_title = bool(target_named & title_terms)
        shared_platform = bool(
            platforms & named_title_anchors(row["title"])
            & named_title_anchors(title) & target_terms
        )
        if not shared_subject_in_title and not shared_platform:
            continue
        anchor_overlap = anchors & article_terms
        target_overlap = target_terms & article_terms
        if len(anchor_overlap) < 2 or len(target_overlap) < 2:
            continue
        if not (anchors & title_terms) and len(anchor_overlap) < 3:
            continue
        index = max(
            range(len(candidate_sentences)),
            key=lambda i: len(news["sentence_terms"][i] & (anchors | target_terms)),
        )
        evidence_sentence = candidate_sentences[index]
        if len(news["sentence_terms"][index] & (anchors | target_terms)) < 2:
            continue
        score = len(anchor_overlap) + len(target_overlap) + len(anchors & title_terms)
        ranked.append((score, news["news_id"], title, source, evidence_sentence))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[:max_items]


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
    scores = {}
    skipped_sections = Counter()

    for section in HEADINGS:
        field = SECTION_FIELDS[section]

        target = build_section_target(
            row.get(field),
            section,
        )

        try:
            selected = select_evidence(
                row=row,
                target_section=section,
                target_text=target,
                keywords=keywords,
            )
        except ValueError:
            skipped_sections["low_evidence_support_" + section.lower()] += 1
            continue

        evidence = [sentence for sentence, _ in selected]
        evidence_terms = content_terms(" ".join(evidence))
        supported_sentences = [
            sentence for sentence in sentences(target)
            if len(content_terms(sentence) & evidence_terms) >= 2
            and len(content_terms(sentence) & evidence_terms)
            / max(1, len(content_terms(sentence))) >= 0.16
        ]
        target = " ".join(supported_sentences)
        if len(target.split()) < TARGET_WORD_LIMITS[section][0]:
            skipped_sections["low_evidence_support_" + section.lower()] += 1
            continue
        score = support_score(target, evidence)
        if score < 0.16:
            skipped_sections["low_evidence_support_" + section.lower()] += 1
            continue

        # 마지막 안전 검사:
        # Target과 Evidence가 완전히 동일한 문장을 공유하는지 확인
        if any(is_near_copy(item, sentence) for item in evidence for sentence in supported_sentences):
            raise ValueError(
                "target_evidence_overlap_"
                + section.lower()
            )

        targets[section] = target
        evidence_by_section[section] = selected
        scores[section] = round(score, 4)

    if not targets:
        raise ValueError("low_evidence_support_all")

    return {
        "row": row,
        "targets": targets,
        "evidence": evidence_by_section,
        "support_scores": scores,
        "skipped_sections": dict(skipped_sections),
    }


def build_samples(
    rows,
    seed=42,
    news_rows=None,
    tokenizer=None,
    max_input_length=384,
    max_target_length=384,
):
    """
    논문 단위로 split한 뒤,
    한 논문에서 Introduction / Body / Conclusion
    총 3개의 학습 sample을 생성한다.
    """
    rejected = Counter()
    eligible = []
    prepared_news = prepare_news_rows(news_rows)

    for row in rows:
        try:
            eligible.append(
                prepare_paper(row)
            )
        except ValueError as exc:
            rejected[str(exc)] += 1

    skipped_sections = Counter()
    for entry in eligible:
        skipped_sections.update(entry["skipped_sections"])

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
    seen_targets = set()
    input_over_budget = 0
    target_over_budget = 0
    input_tokens = {section: [] for section in HEADINGS}
    target_tokens = {section: [] for section in HEADINGS}
    news_by_section = Counter()

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

        pending = []
        failed_reason = None
        pending_input_tokens = {}
        pending_target_tokens = {}
        for section in HEADINGS:
            if section not in representative["targets"]:
                continue
            selected = representative["evidence"][section]
            news = select_news_evidence(row, representative["targets"][section], prepared_news)
            paper_items = [
                format_evidence_item("PAPER", index, title=row["title"], evidence=sentence)
                for index, (sentence, _) in enumerate(selected, start=1)
            ]
            news_items = [
                format_evidence_item("NEWS", index, title=title, source=source, evidence=sentence)
                for index, (_, _, title, source, sentence) in enumerate(news, start=1)
            ]
            value = make_input(
                title=row["title"],
                topic=(
                    ", ".join(keywords[:4])
                    or row["title"]
                ),
                paper_evidence=paper_items,
                news_evidence=news_items,
                section=section,
            )
            target = representative["targets"][section]
            target_key = normalized(target)
            if target_key in seen_targets or any(normalized(item["target"]) == target_key for item in pending):
                failed_reason = "duplicate_target"
                break
            if tokenizer is not None:
                raw_input_len = len(tokenizer.encode(render_prompt(value), add_special_tokens=True))
                raw_target_len = len(tokenizer.encode(target, add_special_tokens=True))
                input_over_budget += int(raw_input_len > max_input_length)
                target_over_budget += int(raw_target_len > max_target_length)
                if raw_target_len > max_target_length:
                    failed_reason = "target_over_budget"
                    break
                try:
                    encoded = encode_input(tokenizer, value, max_input_length)
                    encode_target(tokenizer, target, max_target_length, section=section)
                except ValueError as exc:
                    failed_reason = "input_contract_" + str(exc).split(";")[0]
                    break
                decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
                payload_count = decoded.count("Evidence:") - int("Paper Evidence:" in decoded) - int("News Evidence:" in decoded)
                if payload_count < 1 + int(bool(news)):
                    failed_reason = "missing_encoded_evidence_body"
                    break
                pending_input_tokens[section] = len(encoded["input_ids"])
                pending_target_tokens[section] = raw_target_len
            pending.append({
                "paper_id": row["paper_id"],
                "source_paper_id": row["paper_id"],
                "source_paper_ids": source_paper_ids,
                "section": section,
                "title": value["title"],
                "topic": value["topic"],
                "research_question": value["research_question"],
                "paper_evidence": value["paper_evidence"],
                "news_evidence": value["news_evidence"],
                "evidence_source_ids": [
                    {"type": "paper", "id": row["paper_id"], "section": source}
                    for _, source in selected
                ] + [{"type": "news", "id": int(news_id)} for _, news_id, _, _, _ in news],
                "evidence_support_score": representative["support_scores"][section],
                "input": value,
                "target": target,
                "target_kind": "sectional_evidence_supported_v3",
            })

        if failed_reason:
            rejected[failed_reason] += 1
            continue
        output[split].extend(pending)
        for section in pending_input_tokens:
            input_tokens[section].append(pending_input_tokens[section])
            target_tokens[section].append(pending_target_tokens[section])
        seen_targets.update(normalized(sample["target"]) for sample in pending)
        for sample in pending:
            news_by_section[sample["section"]] += int(sample["news_evidence"] != ["[NO_NEWS_EVIDENCE]"])
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
        "low_evidence_support_removed": sum(
            skipped_sections.values()
        ),
        "skipped_sections": dict(skipped_sections),
        "raw_input_over_384": input_over_budget,
        "raw_target_over_384": target_over_budget,
        "mean_input_tokens_by_section": {
            name: round(sum(input_tokens[name]) / max(1, len(input_tokens[name])), 2)
            for name in HEADINGS
        },
        "mean_target_tokens_by_section": {
            name: round(sum(target_tokens[name]) / max(1, len(target_tokens[name])), 2)
            for name in HEADINGS
        },
        "news_by_section": dict(news_by_section),
        "duplicate_targets_removed": rejected["duplicate_target"],

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
    targets.append(output_dir / "dataset_statistics.json")

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
        "schema_version": 3,

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
            "Evidence is selected by lexical support, not human entailment review. "
            "News is attached only after a distinctive topic/target overlap check. "
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

    all_samples = [sample for split in SPLITS for sample in samples[split]]
    source_ids = {
        split: {
            str(source_id)
            for sample in samples[split]
            for source_id in sample["source_paper_ids"]
        }
        for split in SPLITS
    }
    overlaps = {
        f"{left}_vs_{right}": sorted(source_ids[left] & source_ids[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    news_count = sum(sample["news_evidence"] != ["[NO_NEWS_EVIDENCE]"] for sample in all_samples)
    statistics = {
        "source_papers": report["source_papers"],
        "eligible_papers": report["eligible_papers"],
        "total_samples": len(all_samples),
        "section_counts": dict(Counter(sample["section"] for sample in all_samples)),
        "split_counts": {split: len(samples[split]) for split in SPLITS},
        "paper_evidence_samples": sum(bool(sample["paper_evidence"]) for sample in all_samples),
        "news_evidence_samples": news_count,
        "no_news_evidence_samples": len(all_samples) - news_count,
        "news_by_section": report["news_by_section"],
        "mean_input_tokens_by_section": report["mean_input_tokens_by_section"],
        "mean_target_tokens_by_section": report["mean_target_tokens_by_section"],
        "raw_input_over_384": report["raw_input_over_384"],
        "raw_target_over_384": report["raw_target_over_384"],
        "duplicate_targets_removed": report["duplicate_targets_removed"],
        "very_short_targets_under_20_words": sum(len(sample["target"].split()) < 20 for sample in all_samples),
        "low_evidence_support_removed": report["low_evidence_support_removed"],
        "rejection_reasons": report["rejected"],
        "split_source_paper_id_overlap": overlaps,
        "split_leakage_detected": any(overlaps.values()),
    }
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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

    parser.add_argument("--news-json", type=Path)
    parser.add_argument("--tokenizer-path", type=str, default="google/flan-t5-small")

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

        if args.news_json:
            news_rows = json.loads(args.news_json.read_text(encoding="utf-8-sig"))
        else:
            news_rows = export_news_postgres(args.env_file)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        samples, report = build_samples(
            rows, args.seed, news_rows=news_rows, tokenizer=tokenizer,
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
