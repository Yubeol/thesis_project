from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


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

    @field_validator("title_ko")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "논문 제목을 입력해주세요."
            )

        return value

    @field_validator("topic_ko")
    @classmethod
    def validate_topic(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None


class DraftResponse(BaseModel):
    title: str
    introduction: str
    body: str
    conclusion: str


class SourceResponse(BaseModel):
    type: Literal[
        "paper",
        "news",
    ]
    title: str
    url: str


class GenerateResponse(BaseModel):
    status: Literal[
        "completed",
        "abstained",
    ]

    draft: DraftResponse | None = None

    sources: list[SourceResponse] = Field(
        default_factory=list
    )

    message: str | None = None