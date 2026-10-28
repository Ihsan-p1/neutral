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


class DomainEvaluation(BaseModel):
    name: str
    description: str
    macro_f1: float | None
    n_sentences: int | None
    constant_baseline_macro_f1: float | None


class Evaluation(BaseModel):
    model_version: str
    metric: Literal["macro_f1"]
    threshold: float
    domains: list[DomainEvaluation]


class AnalyzeResponse(BaseModel):
    article_score: float
    n_sentences: int
    n_biased: int
    confidence_mean: float
    sentences: list[SentenceResult]
    model_version: str
    warnings: list[Warning_]
    # The served model's measured scores, so no caller shows a score without them.
    evaluation: Evaluation | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    backend: str
