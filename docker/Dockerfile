# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – build / install dependencies with uv
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifests first (cache-friendly layer)
COPY pyproject.toml ./
# uv needs the package source to resolve the project metadata
COPY src/ ./src/

# Install only production dependencies into a virtual environment
RUN uv sync --no-dev --frozen --no-cache

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – lean runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="tcc-etl" \
      org.opencontainers.image.description="High-performance Python ETL pipeline" \
      org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Make sure the venv is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

ENTRYPOINT ["tcc-etl"]
