from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator


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
    def validate_title_ko(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "논문 제목을 입력해주세요."
            )

        return value

    @field_validator("topic_ko")
    @classmethod
    def validate_topic_ko(
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


class VisualSeriesResponse(BaseModel):
    name: str

    values: list[
        int | float
    ]


class ChartVisualResponse(BaseModel):
    kind: Literal[
        "line",
        "bar",
        "pie",
    ]

    title: str

    labels: list[str]

    series: list[
        VisualSeriesResponse
    ]

    unit: str | None = None

    source_index: int = Field(
        ...,
        ge=0,
    )


class TableVisualResponse(BaseModel):
    kind: Literal[
        "table",
    ]

    title: str

    columns: list[str]

    rows: list[
        list[str]
    ]

    source_index: int = Field(
        ...,
        ge=0,
    )


VisualResponse: TypeAlias = Annotated[
    ChartVisualResponse
    | TableVisualResponse,
    Field(discriminator="kind"),
]

class GenerateResponse(BaseModel):
    status: Literal[
        "completed",
        "abstained",
    ]

    draft: DraftResponse | None = None

    character_count: int = Field(
        default=0,
        ge=0,
    )

    sources: list[
        SourceResponse
    ] = Field(
        default_factory=list
    )

    visuals: list[
        VisualResponse
    ] = Field(
        default_factory=list
    )

    message: str | None = None


class DownloadDocxRequest(BaseModel):
    draft: DraftResponse

    sources: list[
        SourceResponse
    ] = Field(
        default_factory=list
    )