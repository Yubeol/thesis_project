from fastapi import (
    APIRouter,
    HTTPException,
)

from backend.app.schemas.generate import (
    GenerateRequest,
    GenerateResponse,
)

from backend.app.services.paper_agent_service import (
    generate_paper,
)


router = APIRouter(
    prefix="/api",
    tags=["paper-generation"],
)


@router.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate_paper_endpoint(
    request: GenerateRequest,
):
    """
    논문 제목/주제를 입력받아
    최종 논문 초안을 생성한다.
    """

    try:
        return generate_paper(
            title_ko=request.title_ko,
            topic_ko=request.topic_ko,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "논문 생성 중 오류가 발생했습니다: "
                f"{exc}"
            ),
        ) from exc