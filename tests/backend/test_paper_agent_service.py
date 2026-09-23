from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.paper_agent_service import _parse_final_draft


@pytest.mark.parametrize(
    "final_text",
    [
        "서론\n첫 단락.\n\n본론\n근거 단락.\n\n결론\n마지막 단락.",
        "# 실제 제목\n\n## 서론:\n첫 단락.\n\n## 본론:\n근거 단락.\n\n## 결론:\n마지막 단락.",
        "Introduction:\n첫 단락.\n\nBody:\n근거 단락.\n\nConclusion:\n마지막 단락.",
    ],
)
def test_parse_final_draft_accepts_agent_heading_formats(final_text):
    draft = _parse_final_draft(final_text, "입력 제목")
    assert draft["introduction"] == "첫 단락."
    assert draft["body"] == "근거 단락."
    assert draft["conclusion"] == "마지막 단락."
    assert draft["title"] == ("실제 제목" if final_text.startswith("# 실제 제목") else "입력 제목")


def test_parse_final_draft_does_not_treat_section_heading_as_title():
    draft = _parse_final_draft("# 서론\n첫 단락.\n# 본론\n근거 단락.\n# 결론\n마지막 단락.", "입력 제목")
    assert draft["title"] == "입력 제목"


def test_parse_final_draft_rejects_missing_section_instead_of_fabricating_it():
    with pytest.raises(RuntimeError, match="conclusion"):
        _parse_final_draft("서론\n첫 단락.\n본론\n근거 단락.", "입력 제목")


def test_generate_endpoint_accepts_actual_agent_plain_headings():
    agent_result = {
        "status": "completed",
        "title": "정리된 제목",
        "final": "서론\n첫 단락.\n\n본론\n근거 단락.\n\n결론\n마지막 단락.",
        "sources": [],
    }
    with patch("backend.app.services.paper_agent_service.run_agent_pipeline", return_value=agent_result):
        response = TestClient(app).post("/api/generate", json={"title_ko": "입력 제목"})
    assert response.status_code == 200
    assert response.json()["draft"] == {
        "title": "정리된 제목",
        "introduction": "첫 단락.",
        "body": "근거 단락.",
        "conclusion": "마지막 단락.",
    }
