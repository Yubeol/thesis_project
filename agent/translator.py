import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@lru_cache
def get_openai_client() -> OpenAI:
    """
    OpenAI Client를 생성한다.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다."
        )

    return OpenAI(api_key=api_key)


def get_chat_model() -> str:
    """
    번역 및 LLM 처리에 사용할 모델명을 반환한다.
    """
    return os.getenv(
        "OPENAI_CHAT_MODEL",
        "gpt-5.6-luna",
    )


def translate_to_english(
    text_ko: str,
) -> str:
    """
    한국어 텍스트를 RAG 검색에 적합한 영어로 번역한다.
    """

    if not text_ko or not text_ko.strip():
        raise ValueError(
            "번역할 한국어 텍스트가 비어 있습니다."
        )

    client = get_openai_client()

    response = client.responses.create(
        model=get_chat_model(),
        instructions=(
            "Translate the user's Korean academic text "
            "into natural academic English. "
            "Preserve the original meaning, terminology, "
            "proper nouns, and research scope exactly. "
            "Do not add explanations or new information. "
            "Return only the translated English text."
        ),
        input=text_ko.strip(),
    )

    translated = response.output_text.strip()

    if not translated:
        raise RuntimeError(
            "영문 번역 결과가 비어 있습니다."
        )

    return translated


def translate_query_to_english(
    title_ko: str,
    topic_ko: str | None = None,
) -> dict:
    """
    사용자에게 입력받은 한국어 제목/주제를
    각각 영어로 변환한다.

    Hybrid RAG에 넘길 query_en도 함께 생성한다.
    """

    if not title_ko or not title_ko.strip():
        raise ValueError(
            "논문 제목이 비어 있습니다."
        )

    title_en = translate_to_english(
        title_ko
    )

    topic_en = None

    if topic_ko and topic_ko.strip():
        topic_en = translate_to_english(
            topic_ko
        )

    if topic_en:
        query_en = (
            f"Research title: {title_en}\n"
            f"Research topic: {topic_en}"
        )
    else:
        query_en = title_en

    return {
        "title_ko": title_ko.strip(),
        "topic_ko": (
            topic_ko.strip()
            if topic_ko
            else None
        ),
        "title_en": title_en,
        "topic_en": topic_en,
        "query_en": query_en,
    }

def translate_to_korean(
    text_en: str,
) -> str:
    """
    영문 학술 텍스트를 자연스러운 한국어 학술문체로 번역한다.
    """

    if not text_en or not text_en.strip():
        raise ValueError(
            "번역할 영문 텍스트가 비어 있습니다."
        )

    client = get_openai_client()

    response = client.responses.create(
        model=get_chat_model(),
        instructions=(
            "Translate the user's English academic text "
            "into natural Korean academic writing. "
            "Preserve the original meaning, terminology, "
            "claims, and scope exactly. "
            "Do not add, remove, or invent information. "
            "Return only the translated Korean text."
        ),
        input=text_en.strip(),
    )

    translated = response.output_text.strip()

    if not translated:
        raise RuntimeError(
            "한국어 번역 결과가 비어 있습니다."
        )

    return translated


def translate_draft_to_korean(
    draft_en: dict,
) -> dict:
    """
    영문 Introduction / Body / Conclusion을
    각각 한국어로 변환한다.
    """

    required = {
        "introduction",
        "body",
        "conclusion",
    }

    missing = required - draft_en.keys()

    if missing:
        raise ValueError(
            "영문 초안에 필수 항목이 없습니다: "
            + ", ".join(sorted(missing))
        )

    return {
        "introduction": translate_to_korean(
            draft_en["introduction"]
        ),
        "body": translate_to_korean(
            draft_en["body"]
        ),
        "conclusion": translate_to_korean(
            draft_en["conclusion"]
        ),
    }