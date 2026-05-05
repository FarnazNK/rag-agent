# syntax=docker/dockerfile:1.7

# ---------- Builder stage ----------
# We use a builder stage so the final image doesn't carry build tools or the
# pip cache. Keeps the production image small and the attack surface narrow.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build deps once, then copy code so dep installs cache on code-only edits.
COPY pyproject.toml README.md ./
COPY src ./src

# Install with API extras (FastAPI, uvicorn, prometheus-client). The `[api]`
# extra is defined in pyproject.toml.
RUN pip install --upgrade pip && \
    pip install --user ".[api]"


# ---------- Runtime stage ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}" \
    APP_VECTOR_STORE_PATH=/data/chroma

# Non-root user — never run prod containers as root.
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /data /app/data && \
    chown -R app:app /data /app

WORKDIR /app

# Pull the user-installed packages from the builder.
COPY --from=builder /root/.local /home/app/.local
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data
COPY --chown=app:app scripts ./scripts

USER app

EXPOSE 8000

# A simple healthcheck so docker (and orchestrators) can detect a wedged
# process. Curl is not in slim images, so we use python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Default command runs the API. Override in compose for one-off jobs (evals).
CMD ["uvicorn", "rag_agent.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
