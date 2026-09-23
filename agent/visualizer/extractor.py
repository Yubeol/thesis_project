import json
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from openai import OpenAI

from agent.prompts.visualizer import VISUALIZER_SYSTEM_PROMPT


MAX_CHUNKS_PER_SOURCE = 3
MAX_CHARS_PER_CHUNK = 3000
MAX_VISUALS = 6

VALID_KINDS = {
    "line",
    "bar",
    "pie",
    "table",
}

# [수정] 맨 앞에 "연구 쟁점과의 관련성" 조건 추가
VISUAL_EXTRACTION_INSTRUCTION = """
[최우선 조건: 연구 쟁점과의 관련성]
모든 visual은 question.title이 묻는 쟁점에 직접 답하는 근거여야 한다.
source에 비교 가능한 내용이 있더라도, 논문 제목의 쟁점과 직접 관련이 없으면
그 source로 visual을 만들지 않는다.
예: 제목이 "지역별 팬덤 분포"라면 지역·국가별 비교가 담긴 근거만 사용하고,
사회 운동 참여 사례처럼 다른 쟁점의 근거는 사용하지 않는다.

각 visual에는 "relevance" 필드를 추가하여,
이 visual이 논문 제목의 쟁점과 어떻게 연결되는지 한 문장으로 적는다.
한 문장으로 연결을 설명할 수 없다면 그 visual은 만들지 않는다.

가능한 경우 서로 다른 source_index를 사용하여
2~3개의 독립적인 visual 후보를 반환하라.

단, 근거가 부족한데 개수를 맞추기 위해
억지로 visual을 만들면 안 된다.

연구 주제와 직접 관련된 비교 가능한 수치가 존재하면
정성적 table보다 line / bar / pie를 우선한다.

동일한 지표의 항목별 수치가 2개 이상이면 bar를 반드시 포함하고,
동일한 지표의 시점별 수치가 2개 이상이면 line을 반드시 포함한다.
단위나 측정 대상이 서로 다른 숫자를 하나의 chart로 묶지 않는다.

검증 가능한 chart를 만들 수 없지만 source에 연구 쟁점과 관련해
서로 비교하거나 요약할 수 있는 근거 문장이 2개 이상이면 table을 최소 1개 반환한다.

하나의 후보가 부족하더라도
다른 source에 유효한 데이터가 있다면
전체 visuals를 빈 배열로 반환하지 않는다.
""".strip()


# [수정] 재시도에서도 관련성 조건 유지
VISUAL_RETRY_INSTRUCTION = """
이전 추출 결과에서 검증을 통과한 visual이 하나도 남지 않았다.

동일한 source chunk를 다시 검토하라.
검색이나 source를 변경하지 말고,
현재 제공된 근거 안에서만 다시 추출한다.

개수를 채우기 위해 논문 제목의 쟁점과 관련 없는 source를 사용하지 않는다.
각 visual에는 제목의 쟁점과의 연결을 한 문장으로 적은 "relevance" 필드를 포함한다.

특히 다음 조건을 다시 확인한다.

- 2개 이상의 항목과 직접 대응하는 수치 또는 순위
- 2개 이상의 시점과 직접 대응하는 값
- 동일한 전체를 구성하는 2개 이상의 명시적 비율
- 수치화할 수 없지만 명시적으로 비교 가능한 2개 이상의 항목

가능하면 서로 다른 source_index에서
2~3개의 독립적인 visual 후보를 반환한다.

수치 비교가 가능하면 table 대신 chart를 우선한다.

그래도 충분한 근거가 없다면 반드시 다음을 반환한다.

{
  "visuals": []
}
""".strip()

NUMBER_PATTERN = re.compile(
    r"[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?"
)

TOP_N_PATTERN = re.compile(
    r"(?:top\s*(\d+)|상위\s*(\d+))",
    re.IGNORECASE,
)

def _title_exceeds_item_count(
    *,
    title: str,
    item_count: int,
) -> bool:
    """
    제목이 Top 10 / 상위 10처럼
    실제 데이터 개수보다 큰 범위를 주장하는지 확인한다.
    """

    match = TOP_N_PATTERN.search(
        title or ""
    )

    if not match:
        return False

    claimed_text = (
        match.group(1)
        or match.group(2)
    )

    if not claimed_text:
        return False

    claimed_count = int(
        claimed_text
    )

    return claimed_count > item_count


def _paper_url(
    paper: dict[str, Any],
) -> str:
    url = str(
        paper.get("source_url") or ""
    ).strip()

    if url:
        return url

    doi = str(
        paper.get("doi") or ""
    ).strip()

    if not doi:
        return ""

    doi = doi.removeprefix(
        "doi:"
    ).strip()

    if doi.startswith(
        ("http://", "https://")
    ):
        return doi

    return f"https://doi.org/{doi}"


def _news_url(
    news: dict[str, Any],
) -> str:
    return str(
        news.get("url") or ""
    ).strip()


def _build_source_index_map(
    sources: list[dict[str, Any]],
) -> dict[tuple[str, str], int]:
    mapping: dict[
        tuple[str, str],
        int,
    ] = {}

    for index, source in enumerate(
        sources
    ):
        source_type = str(
            source.get("type") or ""
        )

        url = str(
            source.get("url") or ""
        ).strip()

        if source_type and url:
            mapping[
                (source_type, url)
            ] = index

    return mapping


def _collect_visual_sources(
    *,
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    final_retrieval의 실제 chunk content를
    Frontend sources 순서와 연결한다.
    """

    source_index_map = (
        _build_source_index_map(
            sources
        )
    )

    grouped: dict[
        int,
        dict[str, Any],
    ] = {}

    def add_item(
        *,
        source_type: str,
        item: dict[str, Any],
        url: str,
    ) -> None:
        if not url:
            return

        source_index = (
            source_index_map.get(
                (
                    source_type,
                    url,
                )
            )
        )

        if source_index is None:
            return

        content = str(
            item.get("content") or ""
        ).strip()

        if not content:
            return

        if source_index not in grouped:
            grouped[source_index] = {
                "source_index":
                    source_index,

                "type":
                    source_type,

                "title":
                    sources[
                        source_index
                    ].get(
                        "title",
                        "",
                    ),

                "chunks": [],
            }

        chunks = grouped[
            source_index
        ]["chunks"]

        trimmed = content[
            :MAX_CHARS_PER_CHUNK
        ]

        if (
            trimmed not in chunks
            and len(chunks)
            < MAX_CHUNKS_PER_SOURCE
        ):
            chunks.append(
                trimmed
            )

    for paper in retrieval.get(
        "papers",
        [],
    ):
        add_item(
            source_type="paper",
            item=paper,
            url=_paper_url(
                paper
            ),
        )

    for news in retrieval.get(
        "news",
        [],
    ):
        add_item(
            source_type="news",
            item=news,
            url=_news_url(
                news
            ),
        )

    ordered = [
        grouped[index]
        for index in sorted(grouped)
        if grouped[index]["chunks"]
    ]

    papers = [
        source
        for source in ordered
        if source["type"] == "paper"
    ]

    news = [
        source
        for source in ordered
        if source["type"] == "news"
    ]

    selected = (
            papers[:12]
            + news[:12]
    )

    selected.sort(
        key=lambda item:
        item["source_index"]
    )

    return selected


def _safe_json_loads(
    text: str,
) -> dict[str, Any]:
    cleaned = (
        text
        or ""
    ).strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    parsed = json.loads(
        cleaned
    )

    if not isinstance(
        parsed,
        dict,
    ):
        return {
            "visuals": []
        }

    return parsed


def _decimal_from_number(
    value: Any,
) -> Decimal | None:
    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        if isinstance(
            value,
            (int, float),
        ):
            return Decimal(
                str(value)
            )

        if isinstance(
            value,
            str,
        ):
            cleaned = (
                value
                .replace(",", "")
                .strip()
            )

            return Decimal(
                cleaned
            )

    except (
        InvalidOperation,
        ValueError,
    ):
        return None

    return None


def _numbers_in_text(
    text: str,
) -> set[Decimal]:
    values: set[Decimal] = set()

    for match in NUMBER_PATTERN.findall(
        text or ""
    ):
        value = (
            _decimal_from_number(
                match.replace(
                    ",",
                    "",
                )
            )
        )

        if value is not None:
            values.add(
                value
            )

    return values


def _numbers_from_value(
    value: Any,
) -> list[Decimal]:
    if isinstance(
        value,
        bool,
    ):
        return []

    if isinstance(
        value,
        (int, float),
    ):
        decimal_value = (
            _decimal_from_number(
                value
            )
        )

        return (
            [decimal_value]
            if decimal_value
            is not None
            else []
        )

    if isinstance(
        value,
        str,
    ):
        return list(
            _numbers_in_text(
                value
            )
        )

    return []


def _visual_numbers(
    visual: dict[str, Any],
) -> list[Decimal]:
    numbers: list[Decimal] = []

    numbers.extend(
        _numbers_from_value(
            visual.get(
                "title",
                "",
            )
        )
    )

    for label in visual.get(
        "labels",
        [],
    ):
        numbers.extend(
            _numbers_from_value(
                label
            )
        )

    for series in visual.get(
        "series",
        [],
    ):
        if not isinstance(
            series,
            dict,
        ):
            continue

        for value in series.get(
            "values",
            [],
        ):
            numbers.extend(
                _numbers_from_value(
                    value
                )
            )

    for row in visual.get(
        "rows",
        [],
    ):
        if not isinstance(
            row,
            list,
        ):
            continue

        for cell in row:
            numbers.extend(
                _numbers_from_value(
                    cell
                )
            )

    return numbers


def _numbers_exist_in_source(
    *,
    visual: dict[str, Any],
    source_text: str,
) -> bool:
    """
    LLM이 반환한 숫자가 실제 source chunk에
    존재하는지 다시 검증한다.
    """

    source_numbers = (
        _numbers_in_text(
            source_text
        )
    )

    visual_numbers = (
        _visual_numbers(
            visual
        )
    )

    return all(
        number in source_numbers
        for number in visual_numbers
    )


def _sanitize_chart(
    visual: dict[str, Any],
) -> dict[str, Any] | None:
    kind = visual.get(
        "kind"
    )

    labels = visual.get(
        "labels"
    )

    series = visual.get(
        "series"
    )

    if not isinstance(
        labels,
        list,
    ):
        return None

    if len(labels) < 2:
        return None

    labels = [
        str(label)
        for label in labels
    ]

    if not isinstance(
        series,
        list,
    ):
        return None

    if not series:
        return None

    cleaned_series = []

    for item in series:
        if not isinstance(
            item,
            dict,
        ):
            return None

        name = str(
            item.get("name") or ""
        ).strip()

        values = item.get(
            "values"
        )

        if (
            not name
            or not isinstance(
                values,
                list,
            )
        ):
            return None

        if len(values) != len(
            labels
        ):
            return None

        cleaned_values = []

        for value in values:
            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                return None

            cleaned_values.append(
                value
            )

        cleaned_series.append(
            {
                "name": name,
                "values":
                    cleaned_values,
            }
        )

    if (
        kind == "pie"
        and len(cleaned_series) != 1
    ):
        return None

    unit = visual.get(
        "unit"
    )

    if unit is not None:
        unit = str(
            unit
        ).strip() or None

    title = str(
        visual.get("title") or ""
    ).strip()

    if _title_exceeds_item_count(
            title=title,
            item_count=len(labels),
    ):
        return None

    return {
        "kind":
            kind,

        "title": title,

        "labels":
            labels,

        "series":
            cleaned_series,

        "unit":
            unit,

        "source_index":
            visual.get(
                "source_index"
            ),
    }


def _sanitize_table(
    visual: dict[str, Any],
) -> dict[str, Any] | None:
    columns = visual.get(
        "columns"
    )

    rows = visual.get(
        "rows"
    )

    # columns 검증
    if (
        not isinstance(
            columns,
            list,
        )
        or not columns
    ):
        return None

    # 최소 2개의 비교 행 필요
    if (
        not isinstance(
            rows,
            list,
        )
        or len(rows) < 2
    ):
        return None

    cleaned_columns = [
        str(column).strip()
        for column in columns
    ]

    if not all(
        cleaned_columns
    ):
        return None

    cleaned_rows = []

    for row in rows:
        if (
            not isinstance(
                row,
                list,
            )
            or len(row)
            != len(cleaned_columns)
        ):
            return None

        cleaned_rows.append(
            [
                str(cell).strip()
                for cell in row
            ]
        )

    # rows를 다 만든 다음 제목 범위 검증
    title = str(
        visual.get("title") or ""
    ).strip()

    if not title:
        return None

    if _title_exceeds_item_count(
        title=title,
        item_count=len(cleaned_rows),
    ):
        return None

    return {
        "kind": "table",
        "title": title,
        "columns":
            cleaned_columns,
        "rows":
            cleaned_rows,
        "source_index":
            visual.get(
                "source_index"
            ),
    }

def _validate_visuals(
    *,
    raw_visuals: Any,
    visual_sources:
        list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(
        raw_visuals,
        list,
    ):
        return []

    source_texts = {
        source[
            "source_index"
        ]:
        "\n\n".join(
            source["chunks"]
        )

        for source
        in visual_sources
    }

    validated: list[
        dict[str, Any]
    ] = []

    for visual in raw_visuals[
        :MAX_VISUALS
    ]:
        if not isinstance(
            visual,
            dict,
        ):
            continue

        kind = visual.get(
            "kind"
        )

        if kind not in VALID_KINDS:
            continue

        source_index = (
            visual.get(
                "source_index"
            )
        )

        if (
            not isinstance(
                source_index,
                int,
            )
            or source_index
            not in source_texts
        ):
            continue

        if kind == "table":
            cleaned = (
                _sanitize_table(
                    visual
                )
            )
        else:
            cleaned = (
                _sanitize_chart(
                    visual
                )
            )

        if not cleaned:
            continue

        if not cleaned.get(
            "title"
        ):
            continue

        if not _numbers_exist_in_source(
            visual=cleaned,
            source_text=
                source_texts[
                    source_index
                ],
        ):
            continue

        validated.append(
            cleaned
        )

    return validated


# [수정] 첫 번째 근거가 아니라 주제 단어가 가장 많이 맞는 근거를 선택
def _grounded_fallback_table(
    *,
    visual_sources: list[dict[str, Any]],
    focus: str,
) -> list[dict[str, Any]]:
    """Build one table from verbatim source sentences when charts are impossible.

    This fallback never invents values or summaries. It only selects complete
    sentences that already exist in one retrieved source, so a qualitative
    paper can still show a useful visual without manufacturing numeric data.

    모든 source의 문장을 먼저 점수화한 뒤, 연구 주제 단어와 가장 많이
    겹치는 source를 고른다. 점수가 같으면 앞쪽(source_index가 작은) source를
    선택하므로 관련 단어가 전혀 겹치지 않을 때는 기존 동작과 같다.
    """
    focus_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9가-힣-]{2,}", focus or "")
    }

    best_candidate: dict[str, Any] | None = None
    best_score = -1

    for source in visual_sources:
        source_text = " ".join(str(chunk).strip() for chunk in source["chunks"])
        sentences = []
        seen: set[str] = set()
        for sentence in re.split(r"(?<=[.!?。])\s+", source_text):
            cleaned = " ".join(sentence.split()).strip()
            key = cleaned.casefold()
            if not 35 <= len(cleaned) <= 280 or key in seen:
                continue
            seen.add(key)
            hits = sum(term in key for term in focus_terms)
            sentences.append((hits, len(sentences), cleaned))

        if len(sentences) < 2:
            continue

        ranked = sorted(sentences, key=lambda item: (-item[0], item[1]))[:3]
        score = sum(hits for hits, _, _ in ranked)

        if score <= best_score:
            continue

        best_score = score
        rows = [
            [label, sentence]
            for label, (_, _, sentence) in zip(
                ("핵심 근거", "비교 근거", "추가 근거"),
                ranked,
            )
        ]
        best_candidate = {
            "kind": "table",
            "title": "연구 주제 관련 근거 비교",
            "columns": ["구분", "원문 근거"],
            "rows": rows,
            "source_index": source["source_index"],
        }

    if best_candidate is None:
        return []

    return _validate_visuals(
        raw_visuals=[best_candidate],
        visual_sources=visual_sources,
    )

def _request_visuals(
    *,
    client: OpenAI,
    model: str,
    payload: dict[str, Any],
    retry: bool = False,
) -> list[dict[str, Any]]:
    """
    동일한 visual source를 사용하여
    Visualization LLM에 시각화 후보를 요청한다.
    """

    messages = [
        {
            "role": "system",
            "content": VISUALIZER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": VISUAL_EXTRACTION_INSTRUCTION,
        },
    ]

    if retry:
        messages.append(
            {
                "role": "user",
                "content": VISUAL_RETRY_INSTRUCTION,
            }
        )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={
            "type": "json_object",
        },
        messages=messages,
    )

    content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    parsed = _safe_json_loads(
        content
    )

    visuals = parsed.get(
        "visuals",
        [],
    )

    if not isinstance(
        visuals,
        list,
    ):
        return []

    return visuals

def extract_visuals(
    *,
    title: str,
    topic: str,
    research_question: str,
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    검색된 실제 청크 원문을 별도 LLM에 전달해
    시각화용 JSON 데이터를 추출한다.

    수치 chart를 우선하며, 1차 결과에 chart가 없으면 동일 source로
    Visualization 추출만 1회 재시도한다. 그래도 chart/table을 만들지
    못하면 실제 원문 문장을 그대로 사용한 비교표를 생성한다.
    """

    visual_sources: list[dict[str, Any]] = []

    try:
        visual_sources = (
            _collect_visual_sources(
                retrieval=retrieval,
                sources=sources,
            )
        )

        if not visual_sources:
            return []

        payload = {
            "question": {
                "title": title,
                "topic": topic,
                "research_question":
                    research_question,
            },
            "sources": visual_sources,
        }

        client = OpenAI(
            api_key=os.getenv(
                "OPENAI_API_KEY"
            )
        )

        model = (
            os.getenv(
                "OPENAI_VISUAL_MODEL"
            )
            or os.getenv(
                "OPENAI_LLM_MODEL"
            )
            or "gpt-4o-mini"
        )

        # -------------------------
        # 1차 Visualization 추출
        # -------------------------

        raw_visuals = _request_visuals(
            client=client,
            model=model,
            payload=payload,
            retry=False,
        )

        validated = _validate_visuals(
            raw_visuals=raw_visuals,
            visual_sources=visual_sources,
        )

        if any(visual["kind"] in {"line", "bar", "pie"} for visual in validated):
            return validated

        # -------------------------
        # 1차 결과가 전부 검증 탈락한 경우
        # 동일 source로 단 1회 재시도
        # -------------------------

        retry_raw_visuals = (
            _request_visuals(
                client=client,
                model=model,
                payload=payload,
                retry=True,
            )
        )

        retry_validated = (
            _validate_visuals(
                raw_visuals=
                    retry_raw_visuals,
                visual_sources=
                    visual_sources,
            )
        )

        combined = retry_validated or validated
        if combined:
            return combined

        return _grounded_fallback_table(
            visual_sources=visual_sources,
            focus=f"{title} {topic} {research_question}",
        )

    except Exception:
        return _grounded_fallback_table(
            visual_sources=visual_sources,
            focus=f"{title} {topic} {research_question}",
        )