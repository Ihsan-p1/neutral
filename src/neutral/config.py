from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Nothing is hardcoded elsewhere."""

    model_config = SettingsConfigDict(env_prefix="NEUTRAL_", extra="ignore")

    # Model backend. "stub" is used before weights exist and in CI.
    model_backend: Literal["stub", "onnx"] = "stub"
    model_dir: str = "models/current"
    model_version: str = "stub-dev"

    # Binary decision threshold and the calibrated temperature.
    decision_threshold: float = 0.5
    temperature: float = 1.0

    # Input limits.
    max_chars: int = 50_000
    max_sentences: int = 300
    min_sentences_warn: int = 5

    # Rate limiting.
    rate_limit_minute: str = "20/minute"
    rate_limit_hour: str = "300/hour"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
