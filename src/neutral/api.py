import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from neutral.classifier import load_classifier
from neutral.config import Settings, get_settings
from neutral.detect import TextTooLargeError
from neutral.logging_setup import configure_logging
from neutral.pipeline import analyze
from neutral.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

log = logging.getLogger("neutral.api")


def create_api(settings: Settings | None = None) -> FastAPI:
    """Build the API.

    Settings are injected rather than read globally so tests can exercise
    limits and backends without mutating process environment.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    api = FastAPI(title="Media Neutrality Classifier", version="0.1.0")
    limiter = Limiter(key_func=get_remote_address)
    api.state.limiter = limiter
    api.state.classifier = None
    api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    def get_classifier():
        """Load the classifier once, on first use."""
        if api.state.classifier is None:
            api.state.classifier = load_classifier(settings)
        return api.state.classifier

    def too_large_handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    api.add_exception_handler(TextTooLargeError, too_large_handler)

    # Sync handler: FastAPI runs it in a threadpool, so CPU-bound inference
    # does not block the event loop. `async def` here would freeze the server
    # for every other caller during inference.
    @api.post("/analyze", response_model=AnalyzeResponse)
    @limiter.limit(settings.rate_limit_minute)
    @limiter.limit(settings.rate_limit_hour)
    def analyze_endpoint(request: Request, payload: AnalyzeRequest) -> AnalyzeResponse:
        started = time.perf_counter()
        result = analyze(
            payload.text,
            get_classifier(),
            settings,
            return_sentences=payload.return_sentences,
        )
        # Metadata only. Article text never reaches the log.
        log.info(
            "analyze",
            extra={
                "request_id": str(uuid.uuid4()),
                "path": "/analyze",
                "n_chars": len(payload.text),
                "n_sentences": result.n_sentences,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": 200,
            },
        )
        return result

    # Deliberately not rate limited: platforms poll these to decide restarts.
    @api.get("/health", response_model=HealthResponse)
    def health_endpoint() -> HealthResponse:
        try:
            clf = get_classifier()
            clf.predict_proba(["Health check sentence."])
        except Exception:
            log.exception("health check failed")
            return HealthResponse(
                status="degraded", model_version="unknown", backend=settings.model_backend
            )
        return HealthResponse(
            status="ok", model_version=clf.version, backend=settings.model_backend
        )

    @api.get("/version")
    def version_endpoint() -> dict:
        return {
            "model_version": get_classifier().version,
            "backend": settings.model_backend,
        }

    return api
