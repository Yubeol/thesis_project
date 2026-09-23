from fastapi.testclient import TestClient

from backend.app.main import app


def test_pdf_download_contains_visual_payload_and_web_style_sections():
    response = TestClient(app).post(
        "/api/download/pdf",
        json={
            "draft": {
                "title": "팬덤 네트워크와 콘텐츠 경쟁력",
                "introduction": "연구 배경을 설명하는 서론입니다.",
                "body": "팬덤 활동과 콘텐츠 특성을 비교하는 본론입니다.",
                "conclusion": "두 요인을 함께 분석해야 한다는 결론입니다.",
            },
            "sources": [{
                "type": "paper",
                "title": "K-pop fandom study",
                "url": "https://example.org/paper",
            }],
            "visuals": [
                {
                    "kind": "bar",
                    "title": "팬덤 참여 비교",
                    "labels": ["공유", "번역"],
                    "series": [{"name": "참여", "values": [40, 60]}],
                    "unit": "%",
                    "source_index": 0,
                },
                {
                    "kind": "table",
                    "title": "정성적 근거 비교",
                    "columns": ["구분", "근거"],
                    "rows": [["팬덤", "공동 활동"], ["콘텐츠", "음악적 특성"]],
                    "source_index": 0,
                },
            ],
            "generated_at": "2026-09-23T10:30:00+09:00",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 20_000
