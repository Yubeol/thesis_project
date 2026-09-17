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

class GapItem(BaseModel):
    claim: str
    reason: str
    evidence_type: str = "paper"


class GapAnalysis(BaseModel):
    needs_additional_retrieval: bool = False
    gaps: list[GapItem] = Field(default_factory=list)
    paper_queries: list[str] = Field(default_factory=list)
    news_queries: list[str] = Field(default_factory=list)