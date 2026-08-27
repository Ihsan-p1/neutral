# Build stage: resolve dependencies into a self-contained venv.
FROM python:3.13-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Runtime stage: no build tooling, no PyTorch, non-root.
FROM python:3.13-slim

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    NEUTRAL_MODEL_BACKEND=stub

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/

USER appuser
EXPOSE 8000

# The port comes from the environment: HF Spaces uses 7860, Cloud Run and Fly
# use 8080. A hardcoded port would lock this image to one platform.
CMD ["sh", "-c", "uvicorn neutral.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
