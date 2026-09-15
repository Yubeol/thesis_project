import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


@lru_cache
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다."
        )

    return OpenAI(api_key=api_key)


def embed_text(text: str) -> list[float]:
    """
    하나의 영문 텍스트를 embedding vector로 변환한다.
    """
    text = text.strip()

    if not text:
        raise ValueError("Embedding할 텍스트가 비어 있습니다.")

    client = get_openai_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    embedding = response.data[0].embedding

    return embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    여러 텍스트를 한 번에 embedding한다.
    """
    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return []

    client = get_openai_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned_texts,
    )

    return [
        item.embedding
        for item in response.data
    ]


def embed_query(query_en: str) -> list[float]:
    """
    RAG 검색용 영문 Query embedding.
    """
    return embed_text(query_en)