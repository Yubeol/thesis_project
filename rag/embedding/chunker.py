import os

import tiktoken


DEFAULT_CHUNK_TOKENS = int(
    os.getenv("RAG_CHUNK_TOKENS", "700")
)

DEFAULT_CHUNK_OVERLAP = int(
    os.getenv("RAG_CHUNK_OVERLAP", "100")
)

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def get_tokenizer():
    """
    현재 embedding model에 맞는 tokenizer를 반환한다.
    """
    try:
        return tiktoken.encoding_for_model(
            EMBEDDING_MODEL
        )
    except KeyError:
        return tiktoken.get_encoding(
            "cl100k_base"
        )


def chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    긴 텍스트를 token 기준으로 나눈다.

    반환:
    [
        {
            "content": "...",
            "token_count": 650,
        }
    ]
    """

    if not text or not text.strip():
        return []

    if chunk_tokens < 1:
        raise ValueError(
            "chunk_tokens는 1 이상이어야 합니다."
        )

    if overlap_tokens < 0:
        raise ValueError(
            "overlap_tokens는 0 이상이어야 합니다."
        )

    if overlap_tokens >= chunk_tokens:
        raise ValueError(
            "overlap_tokens는 chunk_tokens보다 작아야 합니다."
        )

    tokenizer = get_tokenizer()

    tokens = tokenizer.encode(
        text.strip()
    )

    chunks = []

    start = 0

    while start < len(tokens):
        end = min(
            start + chunk_tokens,
            len(tokens),
        )

        chunk_tokens_list = tokens[
            start:end
        ]

        content = tokenizer.decode(
            chunk_tokens_list
        ).strip()

        if content:
            chunks.append(
                {
                    "content": content,
                    "token_count": len(
                        chunk_tokens_list
                    ),
                }
            )

        if end >= len(tokens):
            break

        start = end - overlap_tokens

    return chunks


def chunk_paper(
    paper: dict,
) -> list[dict]:
    """
    논문의 섹션 구조를 유지한 채 chunk를 생성한다.

    대상:
    abstract
    introduction
    body
    conclusion
    """

    sections = [
        ("abstract", paper.get("abstract")),
        (
            "introduction",
            paper.get("introduction"),
        ),
        ("body", paper.get("body")),
        (
            "conclusion",
            paper.get("conclusion"),
        ),
    ]

    results = []
    chunk_index = 0

    for section_name, text in sections:
        section_chunks = chunk_text(
            text or ""
        )

        for chunk in section_chunks:
            results.append(
                {
                    "section": section_name,
                    "chunk_index": chunk_index,
                    "content_en": (
                        chunk["content"]
                    ),
                    "token_count": (
                        chunk["token_count"]
                    ),
                }
            )

            chunk_index += 1

    return results


def chunk_news(
    news: dict,
) -> list[dict]:
    """
    뉴스의 영어 RAG용 본문을 chunk로 나눈다.
    """

    chunks = chunk_text(
        news.get(
            "content_en_for_rag",
            "",
        )
    )

    return [
        {
            "chunk_index": index,
            "content_en": chunk["content"],
            "token_count": (
                chunk["token_count"]
            ),
        }
        for index, chunk in enumerate(
            chunks
        )
    ]