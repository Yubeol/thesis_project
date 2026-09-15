from agent.llm_finalizer import finalize_english_draft

from agent.output_limiter import (
    combine_korean_draft,
    count_korean_chars,
    enforce_korean_char_limit,
)

from agent.translator import (
    translate_draft_to_korean,
    translate_query_to_english,
)

from agent.transformer_adapter import (
    generate_transformer_draft,
)

from rag.hybrid import (
    hybrid_retrieve,
    validate_or_abstain,
)


def run_retrieval_pipeline(
    title_ko: str,
    topic_ko: str | None = None,
) -> dict:
    """
    Retrieval까지만 수행하는 파이프라인.

    1. 한국어 → 영어
    2. Hybrid RAG
    3. Evidence Validation
    """

    translated = translate_query_to_english(
        title_ko=title_ko,
        topic_ko=topic_ko,
    )

    query_en = translated["query_en"]

    retrieval = hybrid_retrieve(
        query_en=query_en,
    )

    decision = validate_or_abstain(
        retrieval
    )

    return {
        "input": {
            "title_ko": translated["title_ko"],
            "topic_ko": translated["topic_ko"],
        },

        "translated": {
            "title_en": translated["title_en"],
            "topic_en": translated["topic_en"],
            "query_en": query_en,
        },

        "retrieval": retrieval,

        "decision": decision,
    }


def run_agent_pipeline(
    title_ko: str,
    topic_ko: str | None = None,
) -> dict:
    """
    논문 생성 Agent 전체 Pipeline.

    1. 한국어 제목/주제 → 영어
    2. Hybrid RAG
    3. Evidence Validation
    4. 근거 부족 → Abstain
    5. Transformer 초안
    6. LLM 영문 최종 초안
    7. 한국어 번역
    8. 제목 추가
    9. 4500자 제한
    """

    # -----------------------------------------
    # 1. Retrieval Pipeline
    # -----------------------------------------

    retrieval_result = run_retrieval_pipeline(
        title_ko=title_ko,
        topic_ko=topic_ko,
    )

    decision = retrieval_result["decision"]

    # -----------------------------------------
    # 2. Evidence 부족 → 즉시 중단
    # -----------------------------------------

    if decision["action"] == "abstain":
        return {
            **retrieval_result,

            "status": "abstained",

            "transformer_draft": None,
            "draft_en": None,
            "draft_ko": None,
            "final_text": None,
            "character_count": 0,

            "message": decision["message"],
        }

    translated = retrieval_result["translated"]

    # -----------------------------------------
    # 3. Transformer
    # -----------------------------------------

    transformer_draft = (
        generate_transformer_draft(
            title=translated["title_en"],
            topic=translated["topic_en"],
            retrieval=retrieval_result[
                "retrieval"
            ],
        )
    )

    # -----------------------------------------
    # 4. LLM Finalizer
    # -----------------------------------------

    draft_en = finalize_english_draft(
        title_en=translated["title_en"],
        topic_en=translated["topic_en"],
        transformer_draft=transformer_draft,
        retrieval=retrieval_result[
            "retrieval"
        ],
    )

    # -----------------------------------------
    # 5. English → Korean
    # -----------------------------------------

    draft_ko = translate_draft_to_korean(
        draft_en
    )

    # 사용자가 입력한 원래 논문 제목 유지
    draft_ko["title"] = title_ko.strip()

    # -----------------------------------------
    # 6. 4500자 제한
    # -----------------------------------------

    draft_ko = enforce_korean_char_limit(
        draft_ko,
        max_chars=4500,
    )

    # -----------------------------------------
    # 7. 최종 출력 문자열 생성
    # -----------------------------------------

    final_text = combine_korean_draft(
        draft_ko
    )

    character_count = count_korean_chars(
        draft_ko
    )

    # -----------------------------------------
    # 8. 반환
    # -----------------------------------------

    return {
        **retrieval_result,

        "status": "completed",

        "transformer_draft": (
            transformer_draft
        ),

        "draft_en": draft_en,

        "draft_ko": draft_ko,

        "final_text": final_text,

        "character_count": character_count,

        "message": None,
    }