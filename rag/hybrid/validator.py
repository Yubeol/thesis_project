import os

from dotenv import load_dotenv

load_dotenv()


def _get_int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_best_similarity(paper: dict) -> float | None:
    """
    논문 하나에서 확인 가능한 가장 높은 similarity를 반환한다.

    - Vector RAG로 직접 검색된 논문:
      vector_similarity 사용

    - Graph RAG로 발견된 논문:
      PostgreSQL에서 다시 가져온 evidence_chunks의
      similarity 사용
    """

    similarities = []

    vector_similarity = paper.get("vector_similarity")

    if vector_similarity is not None:
        similarities.append(
            float(vector_similarity)
        )

    for chunk in paper.get(
        "evidence_chunks",
        [],
    ):
        similarity = chunk.get("similarity")

        if similarity is not None:
            similarities.append(
                float(similarity)
            )

    if not similarities:
        return None

    return max(similarities)


def validate_evidence(
    hybrid_result: dict,
    min_evidence_papers: int | None = None,
    min_relevant_papers: int | None = None,
    min_evidence_chunks: int | None = None,
    min_similarity: float | None = None,
) -> dict:
    """
    Hybrid RAG 결과가 논문 초안을 생성하기에
    충분한 근거를 가지고 있는지 검사한다.

    검증 기준:
    1. 실제 본문 Chunk를 가진 논문 수
    2. similarity 기준을 만족하는 논문 수
    3. 전체 근거 Chunk 수

    뉴스는 보조 근거이므로 생성 가능 여부의
    핵심 기준에서는 제외한다.
    """

    min_evidence_papers = (
        min_evidence_papers
        if min_evidence_papers is not None
        else _get_int_env(
            "RAG_MIN_EVIDENCE_PAPERS",
            3,
        )
    )

    min_relevant_papers = (
        min_relevant_papers
        if min_relevant_papers is not None
        else _get_int_env(
            "RAG_MIN_RELEVANT_PAPERS",
            2,
        )
    )

    min_evidence_chunks = (
        min_evidence_chunks
        if min_evidence_chunks is not None
        else _get_int_env(
            "RAG_MIN_EVIDENCE_CHUNKS",
            4,
        )
    )

    min_similarity = (
        min_similarity
        if min_similarity is not None
        else _get_float_env(
            "RAG_MIN_SIMILARITY",
            0.45,
        )
    )

    papers = hybrid_result.get(
        "papers",
        [],
    )

    news = hybrid_result.get(
        "news",
        [],
    )

    evidence_papers = []
    relevant_papers = []

    total_chunks = 0

    for paper in papers:
        chunks = [
            chunk
            for chunk in paper.get(
                "evidence_chunks",
                [],
            )
            if chunk.get(
                "content"
            )
        ]

        # 실제 본문 근거가 없으면
        # 초안 생성 근거로 인정하지 않음
        if not chunks:
            continue

        total_chunks += len(chunks)

        best_similarity = (
            _get_best_similarity(paper)
        )

        evidence_item = {
            "paper_id": paper.get(
                "paper_id"
            ),
            "title": paper.get(
                "title"
            ),
            "best_similarity": (
                best_similarity
            ),
            "chunk_count": len(chunks),
            "retrieval_sources": (
                paper.get(
                    "retrieval_sources",
                    [],
                )
            ),
        }

        evidence_papers.append(
            evidence_item
        )

        if (
            best_similarity is not None
            and best_similarity
            >= min_similarity
        ):
            relevant_papers.append(
                evidence_item
            )

    reasons = []

    if (
        len(evidence_papers)
        < min_evidence_papers
    ):
        reasons.append(
            (
                "본문 근거가 확보된 논문이 "
                f"{len(evidence_papers)}개로, "
                f"최소 기준 {min_evidence_papers}개보다 "
                "적습니다."
            )
        )

    if (
        len(relevant_papers)
        < min_relevant_papers
    ):
        reasons.append(
            (
                "관련성 기준을 충족한 논문이 "
                f"{len(relevant_papers)}개로, "
                f"최소 기준 {min_relevant_papers}개보다 "
                "적습니다."
            )
        )

    if total_chunks < min_evidence_chunks:
        reasons.append(
            (
                "사용 가능한 논문 근거 Chunk가 "
                f"{total_chunks}개로, "
                f"최소 기준 {min_evidence_chunks}개보다 "
                "적습니다."
            )
        )

    can_generate = len(reasons) == 0

    return {
        "can_generate": can_generate,

        "status": (
            "sufficient"
            if can_generate
            else "insufficient"
        ),

        "reasons": reasons,

        "stats": {
            "retrieved_papers": len(
                papers
            ),
            "evidence_papers": len(
                evidence_papers
            ),
            "relevant_papers": len(
                relevant_papers
            ),
            "evidence_chunks": (
                total_chunks
            ),
            "news_count": len(news),
        },

        "thresholds": {
            "min_evidence_papers": (
                min_evidence_papers
            ),
            "min_relevant_papers": (
                min_relevant_papers
            ),
            "min_evidence_chunks": (
                min_evidence_chunks
            ),
            "min_similarity": (
                min_similarity
            ),
        },

        "papers": evidence_papers,
    }


def validate_or_abstain(
    hybrid_result: dict,
) -> dict:
    """
    Evidence가 충분하면 generate,
    부족하면 abstain을 반환한다.
    """

    validation = validate_evidence(
        hybrid_result
    )

    if validation["can_generate"]:
        return {
            "action": "generate",
            "validation": validation,
            "message": None,
        }

    return {
        "action": "abstain",
        "validation": validation,
        "message": (
            "검색된 근거가 충분하지 않아 "
            "논문 초안을 생성하지 않았습니다. "
            "관련 논문 근거를 추가로 확보하거나 "
            "주제를 더 구체화해 주세요."
        ),
    }