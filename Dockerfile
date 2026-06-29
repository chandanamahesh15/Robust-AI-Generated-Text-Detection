# syntax=docker/dockerfile:1
# ------------------------------------------------------------------------------
# Serving image for the AI-vs-Human detector.
# Deliberately installs ONLY the production deps (requirements.txt) — no torch /
# tensorflow — so the image is small and starts fast. The research models are
# never on the serving path.
# ------------------------------------------------------------------------------

# ---- Stage 1: build wheels for the production deps ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- Stage 2: slim runtime ----
FROM python:3.11-slim AS runtime

# Non-root user for security.
RUN useradd --create-home --uid 1000 appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install pre-built wheels (fast, no compiler in the final image).
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Application code + the trained artifact + config.
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY artifacts/ ./artifacts/

USER appuser
EXPOSE 8000

# Fail the container if the model can't serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
