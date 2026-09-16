from fastapi import (
    APIRouter,
    Response,
    status,
)

from backend.app.services.health_service import (
    check_dependencies,
)


router = APIRouter(
    tags=["health"],
)


@router.get("/health")
def health_check():
    """
    FastAPI 서버 자체 상태 확인.
    """
    return {
        "status": "ok",
    }


@router.get("/ready")
def readiness_check(
    response: Response,
):
    """
    PostgreSQL / Neo4j 연결 상태 확인.
    """

    result = check_dependencies()

    if not result["ready"]:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result