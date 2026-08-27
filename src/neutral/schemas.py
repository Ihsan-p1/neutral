from typing import Literal

from pydantic import BaseModel, Field

Warning_ = Literal["non_english", "too_short", "truncated"]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    return_sentences: bool = True


class SentenceResult(BaseModel):
    text: str
    label: Literal["biased", "neutral"]
    prob: float
    start: int
    end: int


class AnalyzeResponse(BaseModel):
    article_score: float
    n_sentences: int
    n_biased: int
    confidence_mean: float
    sentences: list[SentenceResult]
    model_version: str
    warnings: list[Warning_]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    backend: str
