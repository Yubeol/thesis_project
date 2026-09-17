# agent/schemas.py

from typing import Optional

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    allowed: bool
    rejection_reason: Optional[str] = None

    title: str = ""
    topic: str = ""
    research_question: str = ""

    paper_queries: list[str] = Field(default_factory=list)
    news_queries: list[str] = Field(default_factory=list)
    graph_keywords: list[str] = Field(default_factory=list)

    instruction: str = ""