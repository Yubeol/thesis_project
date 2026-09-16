from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health():
    """
    FastAPI 서버 자체 상태 확인.
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_ready_success():
    """
    PostgreSQL / Neo4j가 모두 정상인 경우.
    실제 DB는 호출하지 않고 Mock 처리한다.
    """

    mock_result = {
        "ready": True,
        "postgres": {
            "connected": True,
            "error": None,
        },
        "neo4j": {
            "connected": True,
            "error": None,
        },
    }

    with patch(
        "backend.app.api.health.check_dependencies",
        return_value=mock_result,
    ):
        response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["ready"] is True
    assert data["postgres"]["connected"] is True
    assert data["neo4j"]["connected"] is True


def test_ready_failure():
    """
    DB 중 하나라도 연결되지 않으면
    503을 반환하는지 확인한다.
    """

    mock_result = {
        "ready": False,
        "postgres": {
            "connected": True,
            "error": None,
        },
        "neo4j": {
            "connected": False,
            "error": "connection failed",
        },
    }

    with patch(
        "backend.app.api.health.check_dependencies",
        return_value=mock_result,
    ):
        response = client.get("/ready")

    assert response.status_code == 503

    data = response.json()

    assert data["ready"] is False
    assert data["postgres"]["connected"] is True
    assert data["neo4j"]["connected"] is False


def test_generate_completed():
    """
    논문 생성이 정상 완료되는 경우.
    Agent는 Mock 처리한다.
    """

    mock_result = {
        "status": "completed",
        "draft": {
            "title": (
                "소셜미디어 기반 글로벌 팬덤 활동이 "
                "K-POP의 세계적 확산에 미치는 영향"
            ),
            "introduction": "서론 내용",
            "body": "본론 내용",
            "conclusion": "결론 내용",
        },
        "character_count": 120,
        "message": None,
    }

    with patch(
        "backend.app.api.generate.generate_paper",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/generate",
            json={
                "title_ko": (
                    "소셜미디어 기반 글로벌 팬덤 활동이 "
                    "K-POP의 세계적 확산에 미치는 영향"
                ),
                "topic_ko": None,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["draft"] is not None
    assert data["draft"]["title"] == (
        mock_result["draft"]["title"]
    )
    assert data["draft"]["introduction"] == "서론 내용"
    assert data["draft"]["body"] == "본론 내용"
    assert data["draft"]["conclusion"] == "결론 내용"
    assert data["character_count"] == 120
    assert data["message"] is None


def test_generate_abstained():
    """
    근거 부족으로 Agent가 생성을 중단하는 경우.
    """

    mock_result = {
        "status": "abstained",
        "draft": None,
        "character_count": 0,
        "message": (
            "검색된 근거가 충분하지 않아 "
            "논문 초안을 생성하지 않았습니다."
        ),
    }

    with patch(
        "backend.app.api.generate.generate_paper",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/generate",
            json={
                "title_ko": "근거가 부족한 테스트 주제",
                "topic_ko": None,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "abstained"
    assert data["draft"] is None
    assert data["character_count"] == 0
    assert data["message"] is not None


def test_generate_empty_title():
    """
    제목이 비어 있으면 Pydantic validation으로
    요청을 거절하는지 확인한다.
    """

    response = client.post(
        "/api/generate",
        json={
            "title_ko": "",
            "topic_ko": None,
        },
    )

    assert response.status_code == 422

def test_generate_whitespace_title():
    """
    제목이 공백으로만 구성되어 있으면
    Pydantic validation으로 요청을 거절하는지 확인한다.
    """

    response = client.post(
        "/api/generate",
        json={
            "title_ko": "   ",
            "topic_ko": None,
        },
    )

    assert response.status_code == 422


def test_generate_unexpected_agent_status():
    """
    Agent가 예상하지 못한 status를 반환하면
    서버가 500 오류로 처리하는지 확인한다.
    """

    mock_result = {
        "status": "failed",
        "message": "테스트 오류",
    }

    with patch(
        "backend.app.services.paper_agent_service."
        "run_agent_pipeline",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/generate",
            json={
                "title_ko": "테스트 논문 제목",
                "topic_ko": None,
            },
        )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == (
        "논문 생성 중 내부 오류가 발생했습니다."
    )


def test_generate_missing_draft():
    """
    Agent가 completed 상태를 반환했지만
    최종 draft_ko가 없으면 500 오류로
    처리하는지 확인한다.
    """

    mock_result = {
        "status": "completed",
        "draft_ko": None,
        "character_count": 0,
    }

    with patch(
        "backend.app.services.paper_agent_service."
        "run_agent_pipeline",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/generate",
            json={
                "title_ko": "테스트 논문 제목",
                "topic_ko": None,
            },
        )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == (
        "논문 생성 중 내부 오류가 발생했습니다."
    )