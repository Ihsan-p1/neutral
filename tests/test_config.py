import pytest
from pydantic import ValidationError

from neutral.config import Settings


def test_defaults_are_safe():
    s = Settings()
    assert s.model_backend == "stub"
    assert s.max_chars == 50_000
    assert s.max_sentences == 300
    assert 0.0 < s.decision_threshold < 1.0
    assert s.temperature > 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("NEUTRAL_MAX_CHARS", "1234")
    monkeypatch.setenv("NEUTRAL_MODEL_BACKEND", "onnx")
    s = Settings()
    assert s.max_chars == 1234
    assert s.model_backend == "onnx"


def test_rejects_invalid_backend():
    with pytest.raises(ValidationError):
        Settings(model_backend="nonsense")
