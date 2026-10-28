import inspect

from fastapi.testclient import TestClient

from neutral.api import create_api
from neutral.config import Settings

THREE = "The bill passed. The senator objected. Critics called it reckless."


def client(**settings_kw) -> TestClient:
    return TestClient(create_api(Settings(**settings_kw)))


def test_health_reports_backend():
    r = client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "stub"


def test_analyze_returns_expected_shape():
    r = client().post("/analyze", json={"text": THREE})
    assert r.status_code == 200
    body = r.json()
    assert body["n_sentences"] == 3
    assert len(body["sentences"]) == 3
    assert 0.0 <= body["article_score"] <= 1.0
    assert body["sentences"][0]["label"] in {"biased", "neutral"}


def test_analyze_rejects_empty_text():
    assert client().post("/analyze", json={"text": ""}).status_code == 422


def test_analyze_rejects_oversized_text():
    r = client(max_chars=100).post("/analyze", json={"text": "x" * 200})
    assert r.status_code == 413


def test_version_endpoint():
    r = client().get("/version")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_rate_limit_returns_429():
    c = client(rate_limit_minute="2/minute", rate_limit_hour="100/hour")
    codes = [c.post("/analyze", json={"text": THREE}).status_code for _ in range(3)]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429


def test_health_is_not_rate_limited():
    # Platforms poll /health constantly; a 429 there would trigger restarts.
    c = client(rate_limit_minute="2/minute", rate_limit_hour="100/hour")
    assert all(c.get("/health").status_code == 200 for _ in range(5))


def test_analyze_handler_is_sync():
    # A CPU-bound handler must be sync so FastAPI runs it in a threadpool.
    # `async def` would block the event loop for the duration of inference.
    api = create_api(Settings())
    route = next(r for r in api.routes if getattr(r, "path", None) == "/analyze")
    assert not inspect.iscoroutinefunction(route.endpoint)


def test_analyze_stub_has_null_evaluation():
    # The stub reports whatever version it is given; it must still get no scores.
    body = client(model_version="deberta-babe-f0s3").post("/analyze", json={"text": THREE}).json()
    assert body["evaluation"] is None
