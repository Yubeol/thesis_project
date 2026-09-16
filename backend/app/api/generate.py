import logging

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


logger = logging.getLogger(__name__)


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
        logger.exception(
            "논문 생성 중 오류 발생"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "논문 생성 중 내부 오류가 "
                "발생했습니다."
            ),
        ) from exc