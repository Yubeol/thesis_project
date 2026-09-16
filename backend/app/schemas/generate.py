from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    title_ko: str = Field(
        ...,
        min_length=1,
        description="한국어 논문 제목",
    )

    topic_ko: str | None = Field(
        default=None,
        description="한국어 논문 주제 또는 추가 설명",
    )


class DraftResponse(BaseModel):
    title: str
    introduction: str
    body: str
    conclusion: str


class GenerateResponse(BaseModel):
    status: str

    draft: DraftResponse | None = None

    character_count: int = 0

    message: str | None = None